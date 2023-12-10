import jax
from jax import jit
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from immrax.control import Control, ControlledSystem
from immrax.inclusion import interval, natif, jacif, mjacif, mjacM, Interval
from immrax.inclusion import ut2i, i2ut, i2lu
from immrax.inclusion import Ordering
from immrax.embedding import EmbeddingSystem, InclusionEmbedding
from immrax.utils import d_metzler, d_positive
from jaxtyping import Integer, Float
from typing import Any, List, Literal, Callable, NamedTuple, Union, Tuple, Sequence
from pathlib import Path
from functools import partial
from collections import namedtuple
from jax._src.api import api_boundary

from immrax.system import OpenLoopSystem

class NeuralNetwork (Control, eqx.Module) :
    """NeuralNetwork
    
    A fully connected neural network, that extends immrax.Control and eqx.Module. Loads from a directory.

    Expects the following in the directory inputted:

    - arch.txt file in the format:
        inputlen numneurons activation numneurons activation ... numneurons outputlen

    - if load is True, also expects a model.eqx file, for the weights and biases.
    """
    seq:nn.Sequential
    dir:Path
    out_len:int

    def __init__ (self, dir:Path=None, load:bool=True, key:jax.random.PRNGKey=jax.random.PRNGKey(0)) :
        """Initialize a NeuralNetwork using a directory, of the following form

        Args:
            dir (Path, optional): Directory to load from. Defaults to None.
            load (bool, optional): _description_. Defaults to True.
            key (jax.random.PRNGKey, optional): _description_. Defaults to jax.random.PRNGKey(0).
        """
        Control.__init__(self)
        eqx.Module.__init__(self)

        self.dir = Path(dir)
        mods = []
        self.out_len = None
        with open(self.dir.joinpath('arch.txt')) as f :
            arch = f.read().split()

        inputlen = int(arch[0])

        for a in arch[1:] :
            if a.isdigit() :
                mods.append(nn.Linear(inputlen,int(a),key=key))
                inputlen = int(a)
                self.out_len = int(a)
            else :
                if a.lower() == 'relu' :
                    mods.append(nn.Lambda(jax.nn.relu))
                elif a.lower() == 'sigmoid' :
                    mods.append(nn.Lambda(jax.nn.sigmoid))
                elif a.lower() == 'tanh' :
                    mods.append(nn.Lambda(jax.nn.tanh))

        self.seq = nn.Sequential(mods)

        if load :
            loadpath = self.dir.joinpath('model.eqx')
            self.seq = eqx.tree_deserialise_leaves(loadpath, self.seq)
            # print(f'Successfully loaded model from {loadpath}')

    def save (self) :
        savepath = self.dir.joinpath('model.eqx')
        print(f'Saving model to {savepath}')
        eqx.tree_serialise_leaves(savepath, self.seq)

    def loadnpy (self) :
        import numpy as np

        Ws, bs = np.load(self.dir.joinpath('model.npy'), allow_pickle=True)
        new_leaves = jax.tree_util.tree_leaves(self.seq)
        new_leaves[0] = jnp.array(Ws[0])

        seq = self.seq 
        j = 0
        for i, layer in enumerate(self.seq) :
            if isinstance(layer, nn.Linear) :
                seq = eqx.tree_at(lambda seq: seq[i].weight, seq, Ws[j])
                seq = eqx.tree_at(lambda seq: seq[i].bias, seq, bs[j])
                j += 1
        
        savepath = self.dir.joinpath('model.eqx')
        print(f'Saving model to {savepath}')
        eqx.tree_serialise_leaves(savepath, seq)

    def __call__ (self, x:jax.Array) -> jax.Array :
        return self.seq(x)

    def u(self, t:Union[Integer,Float], x: jax.Array) -> jax.Array :
        """Feedback Control Output of the Neural Network evaluated at x: N(x)."""
        return self(x)

"""
The following code was adapted from ______.
"""

import jax_verify as jv
from jax_verify.src import (
    bound_propagation,
    bound_utils,
    concretization,
    synthetic_primitives,
)
from jax_verify.src.linear import backward_crown, linear_relaxations

def to_jv_interval (x:Interval) -> jv.IntervalBound :
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

class CROWNResult (namedtuple('CROWNResult', ['lC', 'uC', 'ld', 'ud'])) :
    def __call__(self, x:Union[jax.Array, Interval]) -> Interval:
        if isinstance(x, Interval) :
            lCp = jnp.clip(self.lC, 0, jnp.inf)
            lCn = jnp.clip(self.lC, -jnp.inf, 0)
            uCp = jnp.clip(self.uC, 0, jnp.inf)
            uCn = jnp.clip(self.uC, -jnp.inf, 0)
            return interval(
                lCp@x.lower + lCn@x.upper + self.ld,
                uCn@x.lower + uCp@x.upper + self.ud
            )
        elif isinstance(x, jax.Array) :
            return interval(self.lC@x + self.ld, self.uC@x + self.ud)

def crown (f:Callable[..., jax.Array], out_len:int = None) -> Callable[..., CROWNResult] :
    if out_len is None :
        try :
            # If f is a NeuralNetwork object, has this property.
            out_len = f.out_len
        except :
            # raise Exception('need obj or out_len')
            pass
    
    obj = jnp.vstack([jnp.eye(out_len), -jnp.eye(out_len)]) if out_len is not None else None

    def F (
        bound
    ):
        """Run CROWN but return linfuns rather than concretized IntervalBounds.

        Args:
        bound: jax_verify.IntervalBound, bounds on the inputs of the function.
        obj: Tensor describing facets of function's output to bound, automated with out_len
        Returns:
        output_bound: Bounds on the output of the function obtained by FastLin
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
            lC = target_linfuns[0].lin_coeffs[:out_len,:],
            uC = -target_linfuns[0].lin_coeffs[out_len:,:],
            ld = target_linfuns[0].offset[:out_len],
            ud = -target_linfuns[0].offset[out_len:],
        )
        # return target_linfuns

    return F

class FastlinResult (namedtuple('FastlinResult', ['C', 'ld', 'ud'])) :
    def __call__(self, x:Union[jax.Array, Interval]) -> Interval :
        if isinstance(x, Interval) :
            Cp = jnp.clip(self.C, 0, jnp.inf)
            Cn = jnp.clip(self.C, -jnp.inf, 0)
            return interval(
                Cp@x.lower + Cn@x.upper + self.ld,
                Cn@x.lower + Cp@x.upper + self.ud
            )
        elif isinstance(x, jax.Array) :
            c = self.C@x
            return interval(c + self.ld, c + self.ud)

def fastlin (f:Callable[..., jax.Array], out_len:int = None) -> Callable[..., FastlinResult] :
    if out_len is None :
        try :
            # If f is a NeuralNetwork object, has this property.
            out_len = f.out_len
        except :
            # raise Exception('need obj or out_len')
            pass
    
    obj = jnp.vstack([jnp.eye(out_len), -jnp.eye(out_len)]) if out_len is not None else None

    @jit
    @api_boundary
    def F (
        bound
    ):
        """Run CROWN but return linfuns rather than concretized IntervalBounds.

        Args:
        bound: jax_verify.IntervalBound, bounds on the inputs of the function.
        obj: Tensor describing facets of function's output to bound, automated with out_len
        Returns:
        output_bound: Bounds on the output of the function obtained by FastLin
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
            C  = target_linfuns[0].lin_coeffs[:out_len,:],
            ld = target_linfuns[0].offset[:out_len],
            ud = -target_linfuns[0].offset[out_len:],
        )

        # return target_linfuns

    return F

class NNCSystem (ControlledSystem) :
    def __init__(self, olsystem: OpenLoopSystem, control: NeuralNetwork) -> None:
        super().__init__(olsystem, control)

class NNCEmbeddingSystem (EmbeddingSystem) :
    sys:NNCSystem
    sys_mjacM:Callable
    verifier:Callable
    nn_verifier:Literal['crown', 'fastlin']
    nn_locality:Literal['local', 'hybrid']

    def __init__(self, sys:NNCSystem, nn_verifier:Literal['crown', 'fastlin'] = 'fastlin',
                 nn_locality:Literal['local', 'hybrid'] = 'hybrid') -> None:
        self.sys = sys
        self.evolution = sys.evolution
        self.xlen = sys.xlen * 2
        self.sys_mjacM = mjacM(sys.olsystem.f)
        self.nn_verifier = nn_verifier
        self.nn_locality = nn_locality
        if nn_verifier == 'crown' :
            self.verifier = crown(sys.control)
        elif nn_verifier == 'fastlin' :
            self.verifier = fastlin(sys.control)
        else :
            raise NotImplementedError(f'nn_verifier must be one of "crown" or "fastlin", {self.nn_verifier} not supported')

    def E (self, t:Interval, x:jax.Array, w:Interval,
           orderings:Tuple[Ordering] = None, centers:jax.Array|Sequence[jax.Array]|None = None) :

        verifier_res = self.verifier(x)
        uglobal = verifier_res(x)
        
    
    # def Fi (i:int, t:jv.IntervalBound, x:jv.IntervalBound, w:jv.IntervalBound) :
    #     pass

    # HYBRID MIXED-CORNERED
    # @partial(ijit, static_argnums=(0,), static_argnames=['orderings'])
    # def E (self, t:jv.IntervalBound, x:jax.Array, w:jv.IntervalBound,
    #        orderings:Tuple[Ordering] =  None, centers:jax.Array|Sequence[jax.Array]|None = None) :
    #     t = interval(t)
    #     ix = ut2i(x)

    #     global_crown_res = self.net_crown(ix)
    #     uglobal = global_crown_res(ix)
    #     corners = ((t.lower, ix.lower, uglobal.lower, w.lower), (t.upper, ix.upper, uglobal.upper, w.upper))
    #     Mpre = self.sys_mjacM(t, ix, uglobal, w, orderings=orderings, centers=corners)
    #     # Jt, Jx, Ju, Jw = [jv.IntervalBound.from_jittable(Mi) for Mi in Mpre[0]]
    #     # tc, xc, uc, wc = tuple([(a.lower + a.upper)/2 for a in [t, ix, uglobal, w]])

    #     # print(uglobal)
    #     # print(corners)

    #     _x, x_ = i2lu(ix)
    #     _w, w_ = i2lu(w)
    #     _C, C_ = global_crown_res.lC, global_crown_res.uC
    #     _d, d_ = global_crown_res.ld, global_crown_res.ud

    #     _ret, ret_ = [], []

    #     n = self.sys.xlen

    #     for i, (tc, xc, uc, wc) in enumerate(corners) :
    #         fc = self.sys.olsystem.f(tc, xc, uc, wc)
    #         Jt, Jx, Ju, Jw = Mpre[i]
    #         _Jx, J_x = i2lu(Jx)
    #         _Ju, J_u = i2lu(Ju)
    #         _Jw, J_w = i2lu(Jw)

    #         _Bp, _Bn = d_positive(_Ju)
    #         B_p, B_n = d_positive(J_u)

    #         _K = _Bp@_C + _Bn@C_
    #         K_ = B_p@C_ + B_n@_C
    #         _Dp, _Dn = d_positive(_Jw); D_p, D_n = d_positive(J_w)

    #         _H = _Jx + _K
    #         H_ = J_x + K_
    #         _Hp, _Hn = d_metzler(_H); H_p, H_n = d_metzler(H_)

    #         # # Bounding the difference: error dynamics for Holding effects
    #         # # _c = 0
    #         # # c_ = 0
    #         # _Kp, _Kn = d_positive(_K); K_p, K_n = d_positive(K_)
    #         # _c = - _Kn@_e - _Kp@e_
    #         # c_ = - K_n@e_ - K_p@_e
    #         # self.control.step(0, x)
    #         # self.e = self.e + self.sys.t_spec.t_step * self.sys.f(x, self.uj, w)[0].reshape(-1)

    #         _ret.append(_Hp@_x + _Hn@x_ - _Jx@xc - _Ju@uc + _Bp@_d + _Bn@d_
    #                     + _Dp@_w - _Dp@w_ + fc)
            
    #         ret_.append(H_n@_x + H_p@x_ - J_x@xc - J_u@uc + B_n@_d + B_p@d_ 
    #                     - D_p@_w + D_p@w_ + fc)

    #     _ret, ret_ = jnp.array(_ret), jnp.array(ret_)

    #     return jnp.concatenate((jnp.max(_ret,axis=0), jnp.min(ret_, axis=0)))

    # LOCAL MIXED-CORNERED
    @partial(jit, static_argnums=(0,), static_argnames=['orderings'])
    def E (self, t:jv.IntervalBound, x:jax.Array, w:jv.IntervalBound,
           orderings:Tuple[Ordering] =  None, centers:Union[jax.Array,Sequence[jax.Array],None] = None) :
        t = interval(t)
        ix = ut2i(x)

        global_crown_res = self.net_crown(ix)
        uglobal = global_crown_res(ix)
        corners = ((t.lower, ix.lower, uglobal.lower, w.lower), (t.upper, ix.upper, uglobal.upper, w.upper))
        Mpre = self.sys_mjacM(t, ix, uglobal, w, orderings=orderings, centers=corners)
        # Jt, Jx, Ju, Jw = [jv.IntervalBound.from_jittable(Mi) for Mi in Mpre[0]]
        # tc, xc, uc, wc = tuple([(a.lower + a.upper)/2 for a in [t, ix, uglobal, w]])

        # print(uglobal)
        # print(corners)

        _w, w_ = i2lu(w)

        _ret, ret_ = [], []

        n = self.sys.xlen

        for i, (tc, xc, uc, wc) in enumerate(corners) :
            _xi = ut2i(jnp.copy(x).at[i+n].set(x[i]))
            global_crown_res = self.net_crown(_xi)
            _C, C_ = global_crown_res.lC, global_crown_res.uC
            _d, d_ = global_crown_res.ld, global_crown_res.ud
            _x, x_ = i2lu(_xi)

            fc = self.sys.olsystem.f(tc, xc, uc, wc)
            Jt, Jx, Ju, Jw = Mpre[i]
            _Jx, J_x = i2lu(Jx)
            _Ju, J_u = i2lu(Ju)
            _Jw, J_w = i2lu(Jw)

            _Bp, _Bn = d_positive(_Ju)
            B_p, B_n = d_positive(J_u)

            _K = _Bp@_C + _Bn@C_
            K_ = B_p@C_ + B_n@_C
            _Dp, _Dn = d_positive(_Jw); D_p, D_n = d_positive(J_w)

            _H = _Jx + _K
            H_ = J_x + K_
            _Hp, _Hn = d_metzler(_H); H_p, H_n = d_metzler(H_)

            _ret.append(_Hp@_x + _Hn@x_ - _Jx@xc - _Ju@uc + _Bp@_d + _Bn@d_
                        + _Dp@_w - _Dp@w_ + fc)
            
            _xi = ut2i(jnp.copy(x).at[i].set(x[i+n]))
            global_crown_res = self.net_crown(_xi)
            _C, C_ = global_crown_res.lC, global_crown_res.uC
            _d, d_ = global_crown_res.ld, global_crown_res.ud
            _x, x_ = i2lu(_xi)

            fc = self.sys.olsystem.f(tc, xc, uc, wc)
            Jt, Jx, Ju, Jw = Mpre[i]
            _Jx, J_x = i2lu(Jx)
            _Ju, J_u = i2lu(Ju)
            _Jw, J_w = i2lu(Jw)

            _Bp, _Bn = d_positive(_Ju)
            B_p, B_n = d_positive(J_u)

            _K = _Bp@_C + _Bn@C_
            K_ = B_p@C_ + B_n@_C
            _Dp, _Dn = d_positive(_Jw); D_p, D_n = d_positive(J_w)

            _H = _Jx + _K
            H_ = J_x + K_
            _Hp, _Hn = d_metzler(_H); H_p, H_n = d_metzler(H_)

            ret_.append(H_n@_x + H_p@x_ - J_x@xc - J_u@uc + B_n@_d + B_p@d_ 
                        - D_p@_w + D_p@w_ + fc)

        _ret, ret_ = jnp.array(_ret), jnp.array(ret_)

        return jnp.concatenate((jnp.max(_ret,axis=0), jnp.min(ret_, axis=0)))


    # # LOCAL, LOCAL MIXED-CORNERED
    # @partial(ijit, static_argnums=(0,), static_argnames=['orderings'])
    # def E (self, t:jv.IntervalBound, x:jax.Array, w:jv.IntervalBound,
    #        orderings:Tuple[Ordering] =  None, centers:jax.Array|Sequence[jax.Array]|None = None) :
    #     t = interval(t)
    #     ix = ut2i(x)

    #     # global_crown_res = self.net_crown(ix)
    #     # uglobal = global_crown_res(ix)
    #     corners = ((t.lower, ix.lower, w.lower), (t.upper, ix.upper, w.upper))
    #     # Mpre = self.sys_mjacM(t, ix, uglobal, w, orderings=orderings, centers=corners)
    #     # Jt, Jx, Ju, Jw = [jv.IntervalBound.from_jittable(Mi) for Mi in Mpre[0]]
    #     # tc, xc, uc, wc = tuple([(a.lower + a.upper)/2 for a in [t, ix, uglobal, w]])

    #     # print(uglobal)
    #     # print(corners)

    #     _w, w_ = i2lu(w)

    #     _ret, ret_ = [], []

    #     n = self.sys.xlen

    #     for i, (tc, xc, wc) in enumerate(corners) :
    #         _xi = ut2i(jnp.copy(x).at[i+n].set(x[i]))
    #         local_crown_res = self.net_crown(_xi)

    #         _C, C_ = local_crown_res.lC, local_crown_res.uC
    #         _d, d_ = local_crown_res.ld, local_crown_res.ud
    #         _x, x_ = i2lu(_xi)

    #         ulocal = local_crown_res(_xi)
    #         uc = ulocal.lower

    #         Mpre = self.sys_mjacM(t, _xi, ulocal, w, orderings=orderings, centers=((tc,xc,uc,wc),))

    #         fc = self.sys.olsystem.f(tc, xc, uc, wc)
    #         Jt, Jx, Ju, Jw = Mpre[0]
    #         _Jx, J_x = i2lu(Jx)
    #         _Ju, J_u = i2lu(Ju)
    #         _Jw, J_w = i2lu(Jw)

    #         _Bp, _Bn = d_positive(_Ju)
    #         B_p, B_n = d_positive(J_u)

    #         _K = _Bp@_C + _Bn@C_
    #         K_ = B_p@C_ + B_n@_C
    #         _Dp, _Dn = d_positive(_Jw); D_p, D_n = d_positive(J_w)

    #         _H = _Jx + _K
    #         H_ = J_x + K_
    #         _Hp, _Hn = d_metzler(_H); H_p, H_n = d_metzler(H_)

    #         _ret.append(_Hp@_x + _Hn@x_ - _Jx@xc - _Ju@uc + _Bp@_d + _Bn@d_
    #                     + _Dp@_w - _Dp@w_ + fc)
            
    #         _xi = ut2i(jnp.copy(x).at[i].set(x[i+n]))
    #         local_crown_res = self.net_crown(_xi)

    #         _C, C_ = local_crown_res.lC, local_crown_res.uC
    #         _d, d_ = local_crown_res.ld, local_crown_res.ud
    #         _x, x_ = i2lu(_xi)

    #         ulocal = local_crown_res(_xi)
    #         uc = ulocal.upper

    #         Mpre = self.sys_mjacM(t, _xi, ulocal, w, orderings=orderings, centers=((tc,xc,uc,wc),))

    #         fc = self.sys.olsystem.f(tc, xc, uc, wc)
    #         Jt, Jx, Ju, Jw = Mpre[0]
    #         _Jx, J_x = i2lu(Jx)
    #         _Ju, J_u = i2lu(Ju)
    #         _Jw, J_w = i2lu(Jw)

    #         _Bp, _Bn = d_positive(_Ju)
    #         B_p, B_n = d_positive(J_u)

    #         _K = _Bp@_C + _Bn@C_
    #         K_ = B_p@C_ + B_n@_C
    #         _Dp, _Dn = d_positive(_Jw); D_p, D_n = d_positive(J_w)

    #         _H = _Jx + _K
    #         H_ = J_x + K_
    #         _Hp, _Hn = d_metzler(_H); H_p, H_n = d_metzler(H_)

    #         ret_.append(H_n@_x + H_p@x_ - J_x@xc - J_u@uc + B_n@_d + B_p@d_ 
    #                     - D_p@_w + D_p@w_ + fc)

    #     _ret, ret_ = jnp.array(_ret), jnp.array(ret_)

    #     return jnp.concatenate((jnp.max(_ret,axis=0), jnp.min(ret_, axis=0)))

    # FASTLIN CL mjac IMPLEMENTATION
    # @partial(ijit, static_argnums=(0,), static_argnames=['orderings'])
    # def E (self, t:jv.IntervalBound, x:jax.Array, w:jv.IntervalBound,
    #        orderings:Tuple[Ordering] =  None, centers:jax.Array|Sequence[jax.Array]|None = None) :
    #     t = interval(t)
    #     ix = ut2i(x)

    #     global_fastlin_res = self.net_fastlin(ix)
    #     uglobal = global_fastlin_res(ix)
    #     Mpre = self.sys_mjacM(t, ix, uglobal, w, orderings=orderings, centers=centers)
    #     Jt, Jx, Ju, Jw = [jv.IntervalBound.from_jittable(Mi) for Mi in Mpre[0]]
    #     tc, xc, uc, wc = tuple([(a.lower + a.upper)/2 for a in [t, ix, uglobal, w]])

    #     def F (t:jv.IntervalBound, x:jv.IntervalBound, w:jv.IntervalBound) :
    #         return (
    #             # ([Jx] + [Ju]C)[\ulx,\olx]
    #             (Jx + Ju @ global_fastlin_res.C) @ x
    #             # + [Ju][\uld,\old]
    #             + Ju @ interval(global_fastlin_res.ld, global_fastlin_res.ud)
    #             # - [Jx]\overcirc{x} - [Ju]\overcirc{u}
    #             + Jx @ (-xc) + Ju @(-uc)
    #             # + [Jw]([\ulw,\olw] - \overcirc{w})
    #             + Jw @ interval(w.lower - wc, w.upper - wc)
    #             # + f(xc, uc, wc)
    #             + self.sys.olsystem.f(tc, xc, uc, wc)
    #         )

    #     if self.evolution == 'continuous' :
    #         n = self.sys.xlen
    #         ret = jnp.empty(self.xlen)
    #         for i in range(n) :
    #             _xi = jnp.copy(x).at[i+n].set(x[i])
    #             ret = ret.at[i].set(F(interval(t), ut2i(_xi), w).lower[i])
    #             x_i = jnp.copy(x).at[i].set(x[i+n])
    #             ret = ret.at[i+n].set(F(interval(t), ut2i(x_i), w).upper[i])
    #         return ret
    #     elif self.evolution == 'discrete' :
    #         # Convert x from ut to i, compute through F, convert back to ut.
    #         return i2ut(F(t, ix, w))
    #     else :
    #         raise Exception("evolution needs to be 'continuous' or 'discrete'")
