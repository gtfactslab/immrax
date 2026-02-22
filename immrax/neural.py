from collections import namedtuple
from pathlib import Path
from typing import Callable, Literal, Sequence, Tuple, Union

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import Float, Integer

from immrax.control import Control, ControlledSystem
from immrax.embedding import EmbeddingSystem
from immrax.inclusion import (
    Corner,
    Interval,
    Permutation,
    i2lu,
    interval,
    mjacM,
    ut2i,
)
from immrax.system import OpenLoopSystem
from immrax.utils import d_positive, set_columns_from_corner

__all__ = [
    "NeuralNetwork",
    "CROWNResult",
    "FastlinResult",
    "crown",
    "fastlin",
    "NNCSystem",
    "NNCEmbeddingSystem",
]


class NeuralNetwork(eqx.Module, Control):
    """NeuralNetwork

    A fully connected neural network, that extends immrax.Control and eqx.Module. Loads from a directory.

    Expects the following in the directory inputted:

    - arch.txt file in the format:
        inputlen numneurons activation numneurons activation ... numneurons outputlen

    - if load is True, also expects a model.eqx file, for the weights and biases.
    """

    seq: nn.Sequential
    dir: Path = eqx.field(static=True)
    out_len: int = eqx.field(static=True)

    def __init__(
        self,
        dir: Path = None,
        load: bool | Path = True,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        """Initialize a NeuralNetwork using a directory, of the following form

        Parameters
        ----------
        dir : Path, optional
            Directory to load from, by default None
        load : bool | Path, optional
            _description_, by default True
        key : jax.random.PRNGKey, optional
            _description_, by default jax.random.PRNGKey(0)
        """
        Control.__init__(self)
        eqx.Module.__init__(self)

        self.dir = Path(dir)
        mods = []
        self.out_len = None
        with open(self.dir.joinpath("arch.txt")) as f:
            arch = f.read().split()

        inputlen = int(arch[0])

        for a in arch[1:]:
            if a.isdigit():
                mods.append(nn.Linear(inputlen, int(a), key=key))
                inputlen = int(a)
                self.out_len = int(a)
            else:
                if a.lower() == "relu":
                    mods.append(nn.Lambda(jax.nn.relu))
                elif a.lower() == "sigmoid":
                    mods.append(nn.Lambda(jax.nn.sigmoid))
                elif a.lower() == "tanh":
                    # Fixes NaN bug with tanh
                    # mods.append(nn.Lambda(jax.nn.tanh))
                    mods.append(nn.Lambda(lambda x: 2 * jax.nn.sigmoid(2 * x) - 1))
                elif a.lower() == "logsig":
                    mods.append(nn.Lambda(jax.nn.log_sigmoid))
                elif a.lower() == 'softplus' :
                    # jax.nn.softplus uses a custom jvp call
                    # mods.append(nn.Lambda(jax.nn.softplus))
                    # mods.append(nn.Lambda(lambda x : jnp.log(1 + jnp.exp(x))))
                    mods.append(nn.Lambda(lambda x : jnp.log1p(jnp.exp(x))))

        self.seq = nn.Sequential(mods)

        if isinstance(load, bool):
            if load:
                loadpath = self.dir.joinpath("model.eqx")
                self.seq = eqx.tree_deserialise_leaves(loadpath, self.seq)
                print(f"Successfully loaded model from {loadpath}")
        elif isinstance(load, str) or isinstance(load, Path):
            loadpath = Path(load).joinpath("model.eqx")
            self.seq = eqx.tree_deserialise_leaves(loadpath, self.seq)
            print(f"Successfully loaded model from {loadpath}")

    def save(self, verbose=True):
        savepath = self.dir.joinpath("model.eqx")
        if verbose :
            print(f"Saving model to {savepath}...", end="")
        eqx.tree_serialise_leaves(savepath, self.seq)
        if verbose :
            print(" done.")

    # def load (self, path) :
    #     loadpath = Path(path).joinpath('model.eqx')
    #     self.seq = eqx.tree_deserialise_leaves(loadpath, self.seq)
    #     print(f'Successfully loaded model from {loadpath}')

    # def set_dir (self, dir:Path) :
    #     self.dir = Path(dir)

    def loadnpy(self):
        import numpy as np

        Ws, bs = np.load(self.dir.joinpath("model.npy"), allow_pickle=True)
        new_leaves = jax.tree_util.tree_leaves(self.seq)
        new_leaves[0] = jnp.array(Ws[0])

        seq = self.seq
        j = 0
        for i, layer in enumerate(self.seq):
            if isinstance(layer, nn.Linear):
                seq = eqx.tree_at(lambda seq: seq[i].weight, seq, Ws[j])
                seq = eqx.tree_at(lambda seq: seq[i].bias, seq, bs[j])
                j += 1

        savepath = self.dir.joinpath("model.eqx")
        print(f"Saving model to {savepath}")
        eqx.tree_serialise_leaves(savepath, seq)

    def loadzeros(self):
        """Initialize the weights and biases to zero."""
        seq = self.seq
        for i, layer in enumerate(self.seq):
            if isinstance(layer, nn.Linear):
                seq = eqx.tree_at(
                    lambda seq: seq[i].weight, seq, jnp.zeros_like(seq[i].weight)
                )
                seq = eqx.tree_at(
                    lambda seq: seq[i].bias, seq, jnp.zeros_like(seq[i].bias)
                )
        savepath = self.dir.joinpath("model.eqx")
        eqx.tree_serialise_leaves(savepath, seq)
        # self.seq = eqx.tree_deserialise_leaves(savepath, self.seq)
        # self = NeuralNetwork(self.dir, load=True)
        # print(f'Successfully zero initialized model and saved to {savepath}')

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.seq(x)

    def u(self, t: Union[Integer, Float], x: jax.Array) -> jax.Array:
        """Feedback Control Output of the Neural Network evaluated at x: N(x).

        Parameters
        ----------
        t : Union[Integer, Float] :

        x : jax.Array :

        Returns
        -------

        """
        return self(x)


from immrax.inclusion.linbp import LinearBound, linbp


class CROWNResult(namedtuple("CROWNResult", ["lC", "uC", "ld", "ud"])):
    def __call__(self, x: Union[jax.Array, Interval]) -> Interval:
        if isinstance(x, Interval):
            lCp = jnp.clip(self.lC, 0, jnp.inf)
            lCn = jnp.clip(self.lC, -jnp.inf, 0)
            uCp = jnp.clip(self.uC, 0, jnp.inf)
            uCn = jnp.clip(self.uC, -jnp.inf, 0)
            return interval(
                lCp @ x.lower + lCn @ x.upper + self.ld,
                uCn @ x.lower + uCp @ x.upper + self.ud,
            )
        elif isinstance(x, jax.Array):
            return interval(self.lC @ x + self.ld, self.uC @ x + self.ud)


def crown(
    f: Callable[..., jax.Array], out_len: int = None
) -> Callable[..., CROWNResult]:
    lb_fn = linbp(f, relu_mode='adaptive')

    def F(ix: Interval) -> CROWNResult:
        lb = lb_fn(ix)
        return CROWNResult(lC=lb.lA, uC=lb.uA, ld=lb.lb, ud=lb.ub)

    return F


class FastlinResult(namedtuple("FastlinResult", ["C", "ld", "ud"])):
    def __call__(self, x: Union[jax.Array, Interval]) -> Interval:
        if isinstance(x, Interval):
            Cp = jnp.clip(self.C, 0, jnp.inf)
            Cn = jnp.clip(self.C, -jnp.inf, 0)
            return interval(
                Cp @ x.lower + Cn @ x.upper + self.ld,
                Cn @ x.lower + Cp @ x.upper + self.ud,
            )
        elif isinstance(x, jax.Array):
            c = self.C @ x
            return interval(c + self.ld, c + self.ud)

    @property
    def lud(self):
        return interval(self.ld, self.ud)


def fastlin(
    f: Callable[..., jax.Array], out_len: int = None
) -> Callable[..., FastlinResult]:
    lb_fn = linbp(f, relu_mode='same-slope')

    def F(ix: Interval) -> FastlinResult:
        lb = lb_fn(ix)
        return FastlinResult(C=lb.lA, ld=lb.lb, ud=lb.ub)

    return F


class NNCSystem(ControlledSystem):
    def __init__(self, olsystem: OpenLoopSystem, control: NeuralNetwork) -> None:
        super().__init__(olsystem, control)


class NNCEmbeddingSystem(EmbeddingSystem):
    sys: NNCSystem
    sys_mjacM: Callable
    verifier: Callable
    nn_verifier: Literal["crown", "fastlin"]
    nn_locality: Literal["local", "hybrid"]
    M_locality: Literal["local", "hybrid"]

    def __init__(
        self,
        sys: NNCSystem,
        nn_verifier: Literal["crown", "fastlin"] = "crown",
        nn_locality: Literal["local", "hybrid"] = "local",
        M_locality: Literal["local", "hybrid"] = "local",
        sys_mjacM: None | Callable = None,
    ) -> None:
        self.sys = sys
        self.evolution = sys.evolution
        self.xlen = sys.xlen * 2

        # mjacM Transform on open-loop dynamics
        self.sys_mjacM = mjacM(sys.olsystem.f) if sys_mjacM is None else sys_mjacM

        self.nn_verifier = nn_verifier
        self.nn_locality = nn_locality
        self.M_locality = M_locality

        # NN Verifier Transform
        if nn_verifier == "crown":
            self.verifier = crown(sys.control)
        elif nn_verifier == "fastlin":
            self.verifier = fastlin(sys.control)
        else:
            raise NotImplementedError(
                f'nn_verifier must be one of "crown" or "fastlin", {self.nn_verifier} not supported'
            )

    def E(
        self,
        t: Interval,
        x: jax.Array,
        w: Interval,
        permutations: Tuple[Permutation] = None,
        centers: jax.Array | Sequence[jax.Array] | None = None,
        corners: Tuple[Corner] | None = None,
        refine: Callable[[Interval], Interval] | None = None,
        T: Union[jax.Array, None] = None,
        **kwargs,
    ):
        if refine is None:
            refine = lambda x: x

        t = interval(t).atleast_1d()
        ix = refine(ut2i(x))

        n = self.sys.xlen
        p = self.sys.control.out_len
        q = len(w)

        # def F (t, x, w) :
        #     ret = []
        #     for permutation in permutations :
        #         for c in corners :
        #             verifier_res = self.verifier(x)
        #             u = verifier_res(x)
        #             _C = verifier_res.lC; C_ = verifier_res.uC
        #             _d = verifier_res.ld; d_ = verifier_res.ud

        #             Jt, Jx, Ju, Jw = self.sys_mjacM(t, x, u, w, corners=(c,), permutations=(permutation,))[0]

        #             tc = t.lower if c[0] == 0 else t.upper
        #             xc = jnp.array([x[i].lower if c[i+1] == 0 else x[i].upper for i in range(n)])
        #             uc = jnp.array([u[i].lower if c[i+1+n] == 0 else u[i].upper for i in range(p)])
        #             wc = jnp.array([w[i].lower if c[i+1+n+p] == 0 else w[i].upper for i in range(q)])
        #             fc = self.sys.olsystem.f(tc, xc, uc, wc)

        #             return (Jx + Ju@)

        # TODO: Default permutations
        # leninputsfull = tuple([len(x) for x in args])
        # leninputs = sum(leninputsfull)
        # if permutations is None :
        #     permutations = standard_permutation(leninputs)
        # elif isinstance(permutations, Permutation) :
        #     permutations = [permutations]
        # elif not isinstance(permutations, Tuple) :
        #     raise Exception('Must pass jax.Array (one permutation), Sequence[jax.Array], or None (auto standard permutation) for the permutations argument')

        verifier_res = self.verifier(ix)
        uglobal = verifier_res(ix)

        args = (t, ix, uglobal, w)

        n = self.sys.xlen
        p = len(uglobal)
        q = len(w)

        if self.nn_verifier == "crown":
            """ Embedding System induced by Closed-Loop Mixed Cornered Inclusion Function
                For more information, see 'Efficient Interaction-Aware Interval Analysis of Neural Network Feedback Loops'
                https://arxiv.org/pdf/2307.14938.pdf
            """

            if centers is not None:
                raise NotImplementedError(
                    "centers not supported for crown, use cornered mode"
                )
            if corners is None:
                raise Exception("Must pass corners for crown, mixed cornered mode")
            # x0_corners = [tuple([(x.lower if c[i] == 0 else x.upper) for i,x in enumerate(args)]) for c in corners]
            # print(x0_corners)

            txuw_corners = []

            for c in corners:
                tc = t.lower if c[0] == 0 else t.upper
                xc = jnp.array(
                    [ix[i].lower if c[i + 1] == 0 else ix[i].upper for i in range(n)]
                )
                uc = jnp.array(
                    [
                        uglobal[i].lower if c[i + 1 + n] == 0 else uglobal[i].upper
                        for i in range(p)
                    ]
                )
                wc = jnp.array(
                    [
                        w[i].lower if c[i + 1 + n + p] == 0 else w[i].upper
                        for i in range(q)
                    ]
                )
                txuw_corners.append((tc, xc, uc, wc))

            _x = x[:n]
            x_ = x[n:]
            _w, w_ = i2lu(w)

            _ret, ret_ = [], []

            for permutation in permutations:
                # Compute Hybrid M centerings once
                if self.M_locality == "hybrid":
                    Mpre = self.sys_mjacM(
                        t,
                        ix,
                        uglobal,
                        w,
                        permutations=permutation,
                        centers=txuw_corners,
                    )

                for c in corners:
                    # for j, (tc, xc, uc, wc) in enumerate(txuw_corners) :
                    # def body_fun_2 (_, a2) :
                    # tc, xc, uc, wc = txuw_corners[j]
                    # j = i2
                    # tc, xc, uc, wc, c = a2
                    # c = corners[j]
                    # print('here: ', tc, xc, uc, wc)
                    # def body_fun_3 (i3, a3) :
                    def _F(t, x):
                        # LOWER BOUND
                        # _xi = refine(ut2i(jnp.copy(x).at[i3+n].set(x[i3])))
                        _xi = refine(x)

                        # _xc = jnp.minimum(xc, _xi.upper)
                        _xc = jnp.array(
                            [
                                _xi.lower[i] if c[i + 1] == 0 else _xi.upper[i]
                                for i in range(n)
                            ]
                        )

                        # Compute Local NN verification step, otherwise use global
                        if self.nn_locality == "local":
                            verifier_res = self.verifier(_xi)

                        _C, C_ = verifier_res.lC, verifier_res.uC
                        _d, d_ = verifier_res.ld, verifier_res.ud
                        _x, x_ = i2lu(_xi)

                        # Compute Local M centerings, otherwise use precomputed
                        _ui = verifier_res(_xi)
                        _uc = jnp.array(
                            [
                                _ui[k].lower if c[k + 1 + n] == 0 else _ui[k].upper
                                for k in range(p)
                            ]
                        )
                        # uc = self.sys.control.u(t, _xc)

                        if self.M_locality == "local":
                            Jt, Jx, Ju, Jw = self.sys_mjacM(
                                t,
                                _xi,
                                _ui,
                                w,
                                permutations=permutation,
                                centers=((tc, _xc, _uc, wc),),
                            )[0]
                        else:
                            Jt, Jx, Ju, Jw = Mpre[j]

                        _Jx, J_x = set_columns_from_corner(c[1 : n + 1], interval(Jx))
                        _Ju, J_u = set_columns_from_corner(
                            c[n + 1 : n + 1 + p], interval(Ju)
                        )
                        _Jw, J_w = set_columns_from_corner(c[n + 1 + p :], interval(Jw))

                        fc = self.sys.olsystem.f(tc, _xc, _uc, wc)

                        _Bp, _Bn = d_positive(_Ju)
                        B_p, B_n = d_positive(J_u)

                        _K = _Bp @ _C + _Bn @ C_
                        K_ = B_p @ C_ + B_n @ _C
                        # _Dp, _Dn = d_positive(_Jw); D_p, D_n = d_positive(J_w)
                        _Dp = jnp.clip(_Jw, 0, jnp.inf)

                        _H = _Jx + _K
                        H_ = J_x + K_
                        _Hp, _Hn = d_positive(_H)
                        # H_p, H_n = d_metzler(H_)

                        # _ret.append(_Hp@_x + _Hn@x_ - _Jx@xc - _Ju@uc + _Bp@_d + _Bn@d_
                        #             + _Dp@_w - _Dp@w_ + fc)
                        return (
                            _Hp @ _x
                            + _Hn @ x_
                            - _Jx @ _xc
                            - _Ju @ _uc
                            + _Bp @ _d
                            + _Bn @ d_
                            + _Dp @ _w
                            - _Dp @ w_
                            + fc
                        )

                    def F_(t, x):
                        # UPPER BOUND
                        # x_i = refine(ut2i(jnp.copy(x).at[i3].set(x[i3+n])))
                        x_i = refine(x)

                        # x_c = jnp.maximum(xc, x_i.lower)
                        x_c = jnp.array(
                            [
                                x_i.lower[i] if c[i + 1] == 0 else x_i.upper[i]
                                for i in range(n)
                            ]
                        )

                        # Compute Local NN verification step, otherwise use global
                        if self.nn_locality == "local":
                            verifier_res = self.verifier(x_i)

                        _C, C_ = verifier_res.lC, verifier_res.uC
                        _d, d_ = verifier_res.ld, verifier_res.ud
                        _x, x_ = i2lu(x_i)

                        # Compute Local M centerings, otherwise use precomputed
                        u_i = verifier_res(x_i)
                        u_c = jnp.array(
                            [
                                u_i[k].lower if c[k + 1 + n] == 0 else u_i[k].upper
                                for k in range(p)
                            ]
                        )
                        # uc = self.sys.control.u(t, x_c)
                        if self.M_locality == "local":
                            Jt, Jx, Ju, Jw = self.sys_mjacM(
                                t,
                                x_i,
                                u_i,
                                w,
                                permutations=permutation,
                                centers=((tc, x_c, u_c, wc),),
                            )[0]
                        else:
                            Jt, Jx, Ju, Jw = Mpre[j]

                        # _Jx, J_x = i2lu(Jx)
                        # _Ju, J_u = i2lu(Ju)
                        # _Jw, J_w = i2lu(Jw)

                        _Jx, J_x = set_columns_from_corner(c[1 : n + 1], interval(Jx))
                        _Ju, J_u = set_columns_from_corner(
                            c[n + 1 : n + 1 + p], interval(Ju)
                        )
                        _Jw, J_w = set_columns_from_corner(c[n + 1 + p :], interval(Jw))

                        fc = self.sys.olsystem.f(tc, x_c, u_c, wc)

                        _Bp, _Bn = d_positive(_Ju)
                        B_p, B_n = d_positive(J_u)

                        _K = _Bp @ _C + _Bn @ C_
                        K_ = B_p @ C_ + B_n @ _C
                        # _Dp, _Dn = d_positive(_Jw);
                        # D_p, D_n = d_positive(J_w)
                        D_p = jnp.clip(J_w, 0, jnp.inf)

                        _H = _Jx + _K
                        H_ = J_x + K_
                        # _Hp, _Hn = d_metzler(_H);
                        H_p, H_n = d_positive(H_)

                        # ret_.append(H_n@_x + H_p@x_ - J_x@xc - J_u@uc + B_n@_d + B_p@d_
                        #             - D_p@_w + D_p@w_ + fc)
                        return (
                            H_n @ _x
                            + H_p @ x_
                            - J_x @ x_c
                            - J_u @ u_c
                            + B_n @ _d
                            + B_p @ d_
                            - D_p @ _w
                            + D_p @ w_
                            + fc
                        )

                    # Computing F on the faces of the hyperrectangle
                    _X = interval(
                        jnp.tile(_x, (n, 1)),
                        jnp.where(jnp.eye(n), _x, jnp.tile(x_, (n, 1))),
                    )
                    _E = jax.vmap(_F, (None, 0))(t, _X)

                    X_ = interval(
                        jnp.where(jnp.eye(n), x_, jnp.tile(_x, (n, 1))),
                        jnp.tile(x_, (n, 1)),
                    )
                    E_ = jax.vmap(F_, (None, 0))(t, X_)

                    # return jnp.concatenate((jnp.diag(_E), jnp.diag(E_)))
                    _ret.append(jnp.diag(_E))
                    ret_.append(jnp.diag(E_))

            _ret, ret_ = jnp.array(_ret), jnp.array(ret_)

            return jnp.concatenate((jnp.max(_ret, axis=0), jnp.min(ret_, axis=0)))

        # elif self.nn_verifier == 'fastlin' :
        #     """ Embedding System induced by Closed-Loop Mixed Centered Inclusion Function
        #         For more information, see 'Forward Invariance in Neural Network Controlled Systems'
        #         https://arxiv.org/pdf/2309.09043.pdf
        #     """
        #     # TODO: Implement transformation for centered mode using fastlin

        #     # Mixed Centered
        #     if centers is None :
        #         if corners is None or kwargs.get('auto_centered', False) :
        #             # Auto-centered
        #             centers = [tuple([(x.lower + x.upper)/2 for x in args])]
        #         else :
        #             centers = []
        #     elif isinstance(centers, jax.Array) :
        #         centers = [centers]
        #     elif not isinstance(centers, Sequence) :
        #         raise Exception('Must pass jax.Array (one center), Sequence[jax.Array], or None (auto-centered) for the centers argument')

        #     if corners is not None :
        #         if not isinstance(corners, Tuple) :
        #             raise Exception('Must pass Tuple[Corner] or None for the corners argument')
        #         centers.extend([tuple([(x.lower if c[i] == 0 else x.upper) for i,x in enumerate(args)]) for c in corners])

        #     _ret, ret_ = [], []

        #     # for permutation in permutations :V
        #     #     # Compute Hybrid M centerings once
        #     #     if self.M_locality == 'hybrid' or T == 'automatic' :
        #     #         Mpre = self.sys_mjacM(t, ix, uglobal, w, permutations=permutation, centers=txuw_corners)

        #     #     if T == 'automatic' :

        #     #     for i, (tc, xc, uc, wc) in enumerate(centers) :
