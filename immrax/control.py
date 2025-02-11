import abc
import jax
import jax.numpy as jnp
from jaxtyping import Integer, Float
from typing import Union
from sympy2jax import SymbolicModule
from immrax.system import System, OpenLoopSystem

__all__ = [
    'Control',
    'ControlledSystem',
    'LinearControl',
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
