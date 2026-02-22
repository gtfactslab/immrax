"""Benchmark tighten_bounds=True vs False: runtime and output bound width."""
import jax
import jax.numpy as jnp
import equinox.nn as nn
from immrax import interval
from immrax.inclusion.linbp import linbp, _concretize
from immrax.utils import run_times


def build_net(arch, key):
    layers = []
    for i in range(len(arch) - 1):
        key, sk = jax.random.split(key)
        layers.append(nn.Linear(arch[i], arch[i + 1], key=sk))
        if i < len(arch) - 2:
            layers.append(nn.Lambda(jax.nn.relu))
    return nn.Sequential(layers)


def bench(arch, n_rep=100):
    net = build_net(arch, jax.random.PRNGKey(0))
    ix = interval(jnp.full(arch[0], -1.0), jnp.full(arch[0], 1.0))
    name = "-".join(map(str, arch))

    rows = {}
    for tighten in (True, False):
        lb_fn = linbp(net, relu_mode="adaptive", tighten_bounds=tighten)

        # warm-up (JIT compile + run once)
        lb_fn(ix)

        lb, times = run_times(n_rep, lb_fn, ix)
        ms = float(jnp.median(times)) * 1000

        iv = _concretize(lb, ix.lower, ix.upper)
        w = float(jnp.sum(iv.upper - iv.lower))
        rows[tighten] = (ms, w)

    t_ms, t_w = rows[True]
    f_ms, f_w = rows[False]
    improv = (f_w - t_w) / f_w * 100 if f_w > 0 else 0.0
    overhead = (t_ms - f_ms) / f_ms * 100 if f_ms > 0 else 0.0
    print(
        f"{name:24s}  tight={t_ms:6.3f}ms  loose={f_ms:6.3f}ms  "
        f"overhead={overhead:+.0f}%   "
        f"width: {t_w:.4f} -> {f_w:.4f}  tighter_by={improv:.1f}%"
    )


print(f"{'arch':24s}  {'tight':10s}  {'loose':10s}  {'overhead':10s}  bounds comparison")
print("-" * 95)
for arch in [
    [4, 16, 8, 1],
    [4, 32, 32, 4],
    [4, 64, 64, 64, 4],
    [4, 128, 128, 128, 4],
    [8, 64, 64, 64, 64, 8],
    [8, 128, 128, 128, 128, 8],
]:
    bench(arch)
