from collections import namedtuple
from pathlib import Path
from typing import Callable, Literal, Sequence, Tuple, Union

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from jax import jit
from jax._src.traceback_util import api_boundary
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


"""
The following code was adapted from ________.
"""

from . import jax_verify as jv

bound_propagation = jv.src.bound_propagation
bound_utils = jv.src.bound_utils
concretization = jv.src.concretization
synthetic_primitives = jv.src.synthetic_primitives
backward_crown = jv.src.linear.backward_crown
linear_relaxations = jv.src.linear.linear_relaxations


def to_jv_interval(x: Interval) -> jv.IntervalBound:
    return jv.IntervalBound(x.lower, x.upper)


class LinFunExtractionConcretizer(concretization.BackwardConcretizer):
    """Linear function extractor.

    Given an objective over an output, extract the corresponding linear
    function over a target node.
    The relation between the output node and the target node are obtained by
    propagating backward the `base_transform`.
    """

    def __init__(self, base_transform, target_index, obj):
        self._base_transform = base_transform
        self._target_index = target_index
        self._obj = obj

    def should_handle_as_subgraph(self, primitive):
        return self._base_transform.should_handle_as_subgraph(primitive)

    def concretize_args(self, primitive):
        return self._base_transform.concretize_args(primitive)

    def concrete_bound(self, graph, inputs, env, node_ref):
        initial_lin_expression = linear_relaxations.LinearExpression(
            self._obj, jnp.zeros(self._obj.shape[:1])
        )
        target_linfun, _ = graph.backward_propagation(
            self._base_transform,
            env,
            {node_ref: initial_lin_expression},
            [self._target_index],
        )
        return target_linfun


""" 
End of adapted code from ________.
"""


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
    if out_len is None:
        try:
            # If f is a NeuralNetwork object, has this property.
            out_len = f.out_len
        except:
            # raise Exception('need obj or out_len')
            pass

    obj = (
        jnp.vstack([jnp.eye(out_len), -jnp.eye(out_len)])
        if out_len is not None
        else None
    )

    def F(bound):
        """Run CROWN but return linfuns rather than concretized IntervalBounds.

        Parameters
        ----------
        bound :
            Bounds on the inputs of the function.

        Returns
        -------


        """

        bound = to_jv_interval(bound)

        # As we want to extract some linfuns that are in the middle of two linear
        # layers, we want to avoid the linear operator fusion.
        simplifier_composition = synthetic_primitives.simplifier_composition
        default_simplifier_without_linear = simplifier_composition(
            synthetic_primitives.activation_simplifier,
            synthetic_primitives.hoist_constant_computations,
        )

        # We are first going to obtain intermediate bounds for all the activation
        # of the network, so that the backward propagation of the extraction can be
        # done.
        bound_retriever_algorithm = bound_utils.BoundRetrieverAlgorithm(
            concretization.BackwardConcretizingAlgorithm(
                backward_crown.backward_crown_concretizer
            )
        )
        # BoundRetrieverAlgorithm wraps an existing algorithm and captures all of
        # the intermediate bound it generates.
        bound_propagation.bound_propagation(
            bound_retriever_algorithm,
            f,
            bound,
            graph_simplifier=default_simplifier_without_linear,
        )
        intermediate_bounds = bound_retriever_algorithm.concrete_bounds
        # Now that we have extracted all intermediate bounds, we create a
        # FixedBoundApplier. This is a forward transform that pretends to compute
        # bounds, but actually just look them up in a dict of precomputed bounds.
        fwd_bound_applier = bound_utils.FixedBoundApplier(intermediate_bounds)

        # # Let's define what node we are interested in capturing linear functions
        # # for. If needed, this could be extracted and given as argument to this
        # # function, or as a callback that would compute which nodes to target.
        # input_indices = [(i,) for i, _ in enumerate(bounds)]
        # # We're propagating to the first input.
        # target_index = input_indices[0]

        target_index = (0,)

        # Create the concretizer. See the class definition above. The definition
        # of a "concretized_bound" for this one is "Obj linear function
        # reformulated as a linear function of target index".
        # Note: If there is a need to handle a network with multiple output, it
        # should be easy to extend by making obj here a dict mapping output node to
        # objective on that output node.
        extracting_concretizer = LinFunExtractionConcretizer(
            backward_crown.backward_crown_transform, target_index, obj
        )
        # BackwardAlgorithmForwardConcretization uses:
        #  - A forward transform to compute all intermediate bounds (here a bound
        #    applier that just look them up).
        #  - A backward concretizer to compute all final bounds (which we have here
        #    defined as the linear function of the target index).
        fwd_bwd_alg = concretization.BackwardAlgorithmForwardConcretization
        lin_fun_extractor_algorithm = fwd_bwd_alg(
            fwd_bound_applier, extracting_concretizer
        )
        # We get one target_linfuns per output.
        target_linfuns, _ = bound_propagation.bound_propagation(
            lin_fun_extractor_algorithm,
            f,
            bound,
            graph_simplifier=default_simplifier_without_linear,
        )

        return CROWNResult(
            lC=target_linfuns[0].lin_coeffs[:out_len, :],
            uC=-target_linfuns[0].lin_coeffs[out_len:, :],
            ld=target_linfuns[0].offset[:out_len],
            ud=-target_linfuns[0].offset[out_len:],
        )
        # return target_linfuns

    return F


# def crown (net) :
#     def _relu_linprop (l, u) :
#         # branch = jnp.array(l < 0, "int32") + 2*jnp.array(u > 0, "int32")
#         # case0 = lambda : (0., 0., -jnp.inf, jnp.inf) # l > 0, u < 0, Impossible
#         # case1 = lambda : (0., 0., 0., 0.) # l < 0, u < 0, ReLU is inactive
#         # case2 = lambda : (1., 1., 0., 0.) # l < 0, u > 0, ReLU is active
#         # ola = u/(u - l)
#         # case3 = lambda : (ola, 0., -ola*l, 0.) # l < 0, u > 0, ReLU is active.
#         # return lax.switch (branch, (case0, case1, case2, case3))
#         on = l >= 0.
#         active = jnp.logical_and(l < 0., u >= 0.)
#         ua = on.astype(jnp.float32) + active.astype(jnp.float32) * (u / (u - l))
#         ub = active.astype(jnp.float32) * (-ua * l)
#         la = (ua >= 0.5).astype(jnp.float32)
#         lb = jnp.zeros_like(ub)
#         return la, ua, lb, ub
#     def newcrown (ix) :
#         lx = [ix.lower]
#         ux = [ix.upper]

#         # # IBP forward pass

#         # for i, layer in enumerate(net.seq) :
#         #     if isinstance (layer, nn.Linear) :
#         #         # lux = irx.natif(layer)(interval(lx[i], ux[i]))
#         #         # lx.append(lux.lower)
#         #         # ux.append(lux.upper)
#         #         Wp = jnp.clip(layer.weight, 0, jnp.inf); Wn = jnp.clip(layer.weight, -jnp.inf, 0)
#         #         lx.append(Wp @ lx[i] + Wn @ ux[i] + layer.bias)
#         #         ux.append(Wp @ ux[i] + Wn @ lx[i] + layer.bias)

#         #     elif isinstance (layer, nn.Lambda) :
#         #         if layer.fn == jax.nn.relu :
#         #             lx.append(jnp.maximum(lx[i], 0))
#         #             ux.append(jnp.maximum(ux[i], 0))

#         # CROWN forward pass

#         lC = jnp.eye(len(lx[0]))
#         uC = jnp.eye(len(ux[0]))
#         ld = jnp.zeros(len(lx[0]))
#         ud = jnp.zeros(len(ux[0]))

#         for i, layer in enumerate(net.seq) :
#             if isinstance (layer, nn.Linear) :
#                 W = layer.weight
#                 b = layer.bias
#                 Wp, Wn = jnp.clip(W, 0, jnp.inf), jnp.clip(W, -jnp.inf, 0)
#                 _lC = Wp @ lC + Wn @ uC
#                 _uC = Wp @ uC + Wn @ lC
#                 _ld = Wp @ ld + Wn @ ud + b
#                 _ud = Wp @ ud + Wn @ ld + b
#                 _lCp = jnp.clip(_lC, 0, jnp.inf); _lCn = jnp.clip(_lC, -jnp.inf, 0)
#                 _uCp = jnp.clip(_uC, 0, jnp.inf); _uCn = jnp.clip(_uC, -jnp.inf, 0)
#                 # lx.append(jnp.maximum(_lCp @ ix.lower + _lCn @ ix.upper + _ld, Wp @ lx[i] + Wn @ ux[i] + b))
#                 # ux.append(jnp.minimum(_uCp @ ix.upper + _uCn @ ix.lower + _ud, Wp @ ux[i] + Wn @ lx[i] + b))
#                 lx.append(_lCp @ ix.lower + _lCn @ ix.upper + _ld)
#                 ux.append(_uCp @ ix.upper + _uCn @ ix.lower + _ud)
#                 lC, uC, ld, ud = _lC, _uC, _ld, _ud
#             elif isinstance (layer, nn.Lambda) :
#                 if layer.fn == jax.nn.relu :
#                     lCp = jnp.clip(lC, 0, jnp.inf); lCn = jnp.clip(lC, -jnp.inf, 0)
#                     l_con = jnp.maximum(lCp @ ix.lower + lCn @ ix.upper + ld, lx[i])
#                     uCp = jnp.clip(uC, 0, jnp.inf); uCn = jnp.clip(uC, -jnp.inf, 0)
#                     u_con = jnp.minimum(uCp @ ix.upper + uCn @ ix.lower + ud, ux[i])
#                     la, ua, lb, ub = _relu_linprop(l_con, u_con)
#                     lC = (lC.T * la).T
#                     uC = (uC.T * ua).T
#                     ld = lb + la*ld
#                     ud = ub + ua*ud
#                     lx.append(jnp.minimum(lx[i], 0))
#                     ux.append(jnp.maximum(ux[i], 0))

#         # CROWN backward pass

#         lC = jnp.eye(len(lx[-1]))
#         uC = jnp.eye(len(ux[-1]))
#         ld = jnp.zeros(len(lx[-1]))
#         ud = jnp.zeros(len(ux[-1]))

#         for i in range(len(net.seq)-1, -1, -1) :
#             layer = net.seq[i]
#             if isinstance (layer, nn.Linear) :
#                 W = layer.weight
#                 b = layer.bias
#                 _lC = lC @ W
#                 _uC = uC @ W
#                 _ld = lC @ b + ld
#                 _ud = uC @ b + ud
#                 lC, uC, ld, ud = _lC, _uC, _ld, _ud

#             elif isinstance (layer, nn.Lambda) :
#                 if layer.fn == jax.nn.relu :
#                     la, ua, lb, ub = _relu_linprop(lx[i], ux[i])
#                     lCp = jnp.clip(lC, 0, jnp.inf); lCn = jnp.clip(lC, -jnp.inf, 0)
#                     uCp = jnp.clip(uC, 0, jnp.inf); uCn = jnp.clip(uC, -jnp.inf, 0)
#                     _lC = lCp * la + lCn * ua
#                     _uC = uCp * ua + uCn * la
#                     _ld = lCp @ lb + lCn @ ub + ld
#                     _ud = uCp @ ub + uCn @ lb + ud
#                     lC, uC, ld, ud = _lC, _uC, _ld, _ud

#         return CROWNResult(lC, uC, ld, ud)
# return newcrown


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
    if out_len is None:
        try:
            # If f is a NeuralNetwork object, has this property.
            out_len = f.out_len
        except:
            # raise Exception('need obj or out_len')
            pass

    obj = (
        jnp.vstack([jnp.eye(out_len), -jnp.eye(out_len)])
        if out_len is not None
        else None
    )

    @jit
    @api_boundary
    def F(bound):
        """Run CROWN but return linfuns rather than concretized IntervalBounds.

        Parameters
        ----------
        bound :
            bounds on the inputs of the function.

        """

        bound = to_jv_interval(bound)

        # As we want to extract some linfuns that are in the middle of two linear
        # layers, we want to avoid the linear operator fusion.
        simplifier_composition = synthetic_primitives.simplifier_composition
        default_simplifier_without_linear = simplifier_composition(
            synthetic_primitives.activation_simplifier,
            synthetic_primitives.hoist_constant_computations,
        )

        # We are first going to obtain intermediate bounds for all the activation
        # of the network, so that the backward propagation of the extraction can be
        # done.
        bound_retriever_algorithm = bound_utils.BoundRetrieverAlgorithm(
            concretization.BackwardConcretizingAlgorithm(
                backward_crown.backward_fastlin_concretizer
            )
        )

        # BoundRetrieverAlgorithm wraps an existing algorithm and captures all of
        # the intermediate bound it generates.
        bound_propagation.bound_propagation(
            bound_retriever_algorithm,
            f,
            bound,
            graph_simplifier=default_simplifier_without_linear,
        )
        intermediate_bounds = bound_retriever_algorithm.concrete_bounds
        # Now that we have extracted all intermediate bounds, we create a
        # FixedBoundApplier. This is a forward transform that pretends to compute
        # bounds, but actually just look them up in a dict of precomputed bounds.
        fwd_bound_applier = bound_utils.FixedBoundApplier(intermediate_bounds)

        # # Let's define what node we are interested in capturing linear functions
        # # for. If needed, this could be extracted and given as argument to this
        # # function, or as a callback that would compute which nodes to target.
        # input_indices = [(i,) for i, _ in enumerate(bounds)]
        # # We're propagating to the first input.
        # target_index = input_indices[0]

        target_index = (0,)

        # Create the concretizer. See the class definition above. The definition
        # of a "concretized_bound" for this one is "Obj linear function
        # reformulated as a linear function of target index".
        # Note: If there is a need to handle a network with multiple output, it
        # should be easy to extend by making obj here a dict mapping output node to
        # objective on that output node.
        extracting_concretizer = LinFunExtractionConcretizer(
            backward_crown.backward_fastlin_transform, target_index, obj
        )
        # BackwardAlgorithmForwardConcretization uses:
        #  - A forward transform to compute all intermediate bounds (here a bound
        #    applier that just look them up).
        #  - A backward concretizer to compute all final bounds (which we have here
        #    defined as the linear function of the target index).
        fwd_bwd_alg = concretization.BackwardAlgorithmForwardConcretization
        lin_fun_extractor_algorithm = fwd_bwd_alg(
            fwd_bound_applier, extracting_concretizer
        )
        # We get one target_linfuns per output.
        target_linfuns, _ = bound_propagation.bound_propagation(
            lin_fun_extractor_algorithm,
            f,
            bound,
            graph_simplifier=default_simplifier_without_linear,
        )

        return FastlinResult(
            C=target_linfuns[0].lin_coeffs[:out_len, :],
            ld=target_linfuns[0].offset[:out_len],
            ud=-target_linfuns[0].offset[out_len:],
        )

        # return target_linfuns

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
