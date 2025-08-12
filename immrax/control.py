import abc
import jax
import jax.numpy as jnp
from jaxtyping import Integer, Float, ArrayLike
from typing import Union
from immrax.system import System, OpenLoopSystem
from jax.tree_util import register_pytree_node_class

__all__ = [
    'Control',
    'ControlledSystem',
    'LinearControl',
    'InterpControl',
]

class Control (abc.ABC) :
    """Control
    A feedback controller of the form :math:`u:\\mathbb{R}\\times\\mathbb{R}^n\\to\\mathbb{R}^p`.
    """
    @abc.abstractmethod
    def u(self, t:Union[Integer,Float], x:jax.Array) -> jax.Array :
        """Feedback Control Output

        Parameters
        ----------
        t:Union[Integer, Float] :
            
        x:jax.Array :
            

        Returns
        -------

        """


class LinearControl (Control) :
    K: jax.Array
    def __init__(self, K:jax.Array) -> None:
        self.K = K
    def u(self, t:Union[Integer,Float], x:jax.Array) -> jax.Array:
        return self.K @ x

class ControlledSystem (System) :
    """ControlledSystem
    A closed-loop nonlinear dynamical system of the form
    
    .. math::
    
        \\dot{x} = f^{\\textsf{c}}(x,w) = f(x,N(x),w),
    
    where :math:`N:\\mathbb{R}^n \\to \\mathbb{R}^p`.
    """

    olsystem: OpenLoopSystem
    control: Control
    def __init__(self, olsystem:OpenLoopSystem, control:Control) -> None:
        self.olsystem = olsystem
        self.control = control
        self.evolution = olsystem.evolution
        self.xlen = olsystem.xlen
    
    def f (self, t:Union[Integer,Float], x:jax.Array, w:jax.Array) -> jax.Array :
        """Returns the value of the closed loop system

        Parameters
        ----------
        t : Union[Integer, Float]
            time value
        x : jax.Array
            state value
        w : jax.Array
            disturbance value

        Returns
        -------
        jax.Array
            :math:`f^{\\textsf{c}}(x,w) = f(x,N(x),w)`

        """
        # x = jnp.asarray(x); w = jnp.asarray(w)
        return self.olsystem.f(t, x, self.control.u(t, x), w)

# class FOHControlledSystem (ControlledSystem) :
#     """FOHControlledSystem
#     A system in closed-loop with a First Order Hold Controller of the form

#     .. math::

#         \\dot{x} = f^{\\textsf{c}}(x,w) = f(x,N(x(\\tau)),w),
    
#     where :math:`N:\\mathbb{R}^n \\to \\mathbb{R}^p`. The functon `step` is used to set :math:`x(\\tau)`.
#     """

#     ut: jax.Array
#     def __init__(self, system: System, control: Control) -> None:
#         super().__init__(system, control)
    
#     def step(self, x:jax.Array) -> None :
#         self.ut = self.control(x)
    
#     def fc(self, x: jax.Array, w: jax.Array) -> jax.Array:
#         return self.system.f(x, self.ut, w)

@register_pytree_node_class
class InterpControl (Control) :
    t0: Union[Integer, Float]
    tf: Union[Integer, Float]
    us: ArrayLike
    ts: ArrayLike 
    def __init__ (self, t0, tf, us) :
        self.t0 = t0
        self.tf = tf
        # Us should be shape ((tf - t0)/dt, Ulen)
        self.us = us
        self.ts = jnp.linspace(t0, tf, us.shape[0], endpoint=False)

    def tree_flatten (self) :
        return ((self.t0, self.tf, self.us), type(self).__name__)
    
    @classmethod 
    def tree_unflatten (cls, aux_data, children) :
        return cls(*children)
    
# @register_pytree_node_class
# class LinearInterpControl (InterpControl) :
#     def u(self, t, x) :
#         i = jnp.searchsorted(self.ts, t, side='left')

        
