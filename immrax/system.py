import abc
import jax
import jax.numpy as jnp
from jaxtyping import Integer, Float, PyTree
import sympy
from typing import List, Literal, Union, Any, Optional, Callable, Dict, Tuple
from sympy2jax import SymbolicModule
from diffrax import diffeqsolve, ODETerm, Dopri5, Tsit5, Euler, SaveAt, AbstractSolver
from functools import partial

class Trajectory :
    pass

class EvolutionError (Exception) :
    def __init__(self, t:Any, evolution:Literal['continuous','discrete']) -> None:
        super().__init__(f'Time {t} of type {type(t)} does not match evolution type {evolution}')

class System (abc.ABC) :
    """System
    
    An ODE of the form
    
    .. math::
        \\dot{x} = f(t, x, \\dots),
    
    where :math:`t\\in\\T\\in\\{\\mathbb{Z},\\mathbb{R}\\}` is a discrete or continuous time variable, :math:`x\\in\\mathbb{R}^n` is the state of the system, and :math:`\\dots` are some other inputs, perhaps control and disturbance.
    """
    evolution:Literal['continuous', 'discrete']
    xlen:int

    @abc.abstractmethod
    def f (self, t:Union[Integer,Float], x:jax.Array, *args, **kwargs) -> jax.Array :
        """The right hand side of the system

        Parameters
        ----------
        t : Union[Integer, Float]
            The time of the system
        x : jax.Array
            The state of the system
        *args :
            control inputs, disturbance inputs, etc. Depends on parent class.
        **kwargs :
            

        Returns
        -------
        jax.Array
            The time evolution of the state

        """
    
    def fi (self, i:int, t:Union[Integer,Float], x:jax.Array, *args, **kwargs) -> jax.Array :
        """The i-th component of the right hand side of the system

        Parameters
        ----------
        i : int
            component
        t : Union[Integer, Float]
            The time of the system
        x : jax.Array
            The state of the system
        *args :
            control inputs, disturbance inputs, etc. Depends on parent class.
        **kwargs :
            

        Returns
        -------
        jax.Array
            The i-th component of the time evolution of the state

        """
    
        return self.f(t,x,*args,**kwargs)[i]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.f(*args, **kwds)

    # Every argument for jit is static except x0.
    # Thus, recompiled for every new time horizon.
    
    # @partial(jax.jit, static_argnums=(0,1,2,4,5), static_argnames=('f_kwargs',))
    @partial(jax.jit, static_argnums=(0,4), static_argnames=('solver','f_kwargs'))
    def compute_trajectory (self, t0:Union[Integer,Float], tf:Union[Integer,Float], x0:jax.Array, 
                            # inputs:Optional[Dict[str, Callable[[Union[Integer,Float], jax.Array], jax.Array]]] = [],
                            inputs:Tuple[Callable[[int,jax.Array], jax.Array]] = (),
                            dt:float = 0.1, * ,
                            solver:Union[Literal['euler', 'rk45', 'tsit5'],AbstractSolver] = 'tsit5', f_kwargs={}, **kwargs) -> Trajectory :
        def func (t, x, args) :
            # inputargs = {k: v(t, x) for k,v in inputs.items()}
            return self.f(t, x, *[u(t, x) for u in inputs], **f_kwargs)
            # return self.f(t, x, *inputs)
        term = ODETerm(func)
        if solver == 'euler' :
            solver = Euler()
        elif solver == 'rk45' :
            solver = Dopri5()
        elif solver == 'tsit5' :
            solver = Tsit5()
        elif isinstance(solver, AbstractSolver) :
            pass
        else :
            raise Exception(f'{solver=} is not a valid solver')
        # saveat = SaveAt(ts=jnp.arange(t0,tf,dt), t1=True)
        saveat = SaveAt(t0=True, t1=True, steps=True)
        return diffeqsolve(term, solver, t0, tf, dt, x0, saveat=saveat, **kwargs)


class ReversedSystem (System) :
    """ReversedSystem
    A system with time reversed dynamics, i.e. :math:`\\dot{x} = -f(t,x,...)`.
    """
    sys:System
    def __init__(self, sys:System) -> None:
        self.evolution = sys.evolution
        self.xlen = sys.xlen
        self.sys = sys

    def f (self, t:Union[Integer,Float], x:jax.Array, *args, **kwargs) -> jax.Array :
        return -self.sys.f(t,x,*args,**kwargs)

# def revsys (sys:System) :
#     return ReversedSystem(sys)

class OpenLoopSystem (System, abc.ABC) :
    """OpenLoopSystem
    An open-loop nonlinear dynamical system of the form
    
    .. math::
    
        \\dot{x} = f(x,u,w),
    
    where :math:`x\\in\\mathbb{R}^n` is the state of the system, :math:`u\\in\\mathbb{R}^p` is a control input to the system, and :math:`w\\in\\mathbb{R}^q` is a disturbance input.
    """

    @abc.abstractmethod
    def f(self, t:Union[Integer,Float], x:jax.Array, u:jax.Array, w:jax.Array) -> jax.Array :
        """The right hand side of the open-loop system

        Parameters
        ----------
        t : Union[Integer, Float]
            The time of the system
        x : jax.Array
            The state of the system
        u : jax.Array
            The control input to the system
        w : jax.Array
            The disturbance input to the system

        Returns
        -------
        jax.Array
            The time evolution of the state

        """

    def fi(self, i:int, t:Union[Integer,Float], x:jax.Array, u:jax.Array, w:jax.Array) -> jax.Array :
        """The right hand side of the open-loop system

        Parameters
        ----------
        i : int
            component
        t : Union[Integer, Float]
            The time of the system
        x : jax.Array
            The state of the system
        u : jax.Array
            The control input to the system
        w : jax.Array
            The disturbance input to the system

        Returns
        -------
        jax.Array
            The i-th component of the time evolution of the state

        """
        return self.f(t, x, u, w)[i]

class SympySystem (OpenLoopSystem) :
    t_var: sympy.Symbol
    x_vars: sympy.Matrix
    u_vars: sympy.Matrix
    w_vars: sympy.Matrix
    xlen: int
    ulen: int
    wlen: int
    f_eqn: List[sympy.Expr]
    _f: SymbolicModule
    _fi: List[SymbolicModule]

    def __init__(self, t_var:sympy.Symbol, x_vars:List[sympy.Symbol], u_vars:List[sympy.Symbol], w_vars:List[sympy.Symbol],
                 f_eqn:List[sympy.Expr], evolution:Literal['continuous','discrete']='continuous') -> None:
        """Initialize an open-loop system using Sympy.

        Parameters
        ----------
        t_var : sympy.Symbol
            Symbolic variable for the time
        x_vars : List[sympy.Symbol]
            Symbolic variables for the state
        u_vars : List[sympy.Symbol]
            Symbolic variables for the control input
        w_vars : List[sympy.Symbol]
            Symbolic variables for the disturbance input
        f_eqn : List[sympy.Expr]
            Symbolic expressions for the RHS
        evolution : Literal['continuous','discrete'], optional
            Type of time evolution, by default 'continuous'
        """
        self.evolution = evolution

        self.t_var = t_var
        self.x_vars = sympy.Matrix(x_vars)
        self.u_vars = sympy.Matrix(u_vars)
        self.w_vars = sympy.Matrix(w_vars)

        self.xlen = len(x_vars)
        self.ulen = len(u_vars)
        self.wlen = len(w_vars)

        self.f_eqn = [sympy.sympify(f_eqn_i) for f_eqn_i in f_eqn]
        self._f = SymbolicModule(self.f_eqn)
        self._fi = [SymbolicModule(f_eqn_i) for f_eqn_i in self.f_eqn]

    def _txuw_to_kwargs(self, t:Union[Integer,Float], x:jax.Array, u:jax.Array, w:jax.Array) -> dict :
        """Return the kwargs to the SymbolicModule from sympy2jax given a specific value.

        Returns
        -------
        dict
            {name: value} pairs for each symbol and value
        """
        return {
            self.t_var.name: t,
            **{xvar.name: xval for xvar, xval in zip(self.x_vars, x)},
            **{uvar.name: uval for uvar, uval in zip(self.u_vars, u)},
            **{wvar.name: wval for wvar, wval in zip(self.w_vars, w)},
        }
    

    def f (self, t:Union[Integer,Float], x:jax.Array, u:jax.Array, w:jax.Array) -> jax.Array :
        """Get the value of the RHS of the dynamical system.
        
        .. math ::
        
            f (t, x, u, w)

        Parameters
        ----------
        t : Union[Integer, Float]
            state value
        x : jax.Array
            state value
        u : jax.Array
            control value
        w : jax.Array
            disturbance value

        Returns
        -------
        jax.Array
            :math:`f(t,x,u,w)`

        """
        x = jnp.asarray(x); u = jnp.asarray(u); w = jnp.asarray(w)
        return jnp.asarray(self._f(**self._txuw_to_kwargs(t,x,u,w)))

    def fi (self, i:int, t:[Integer,Float], x:jax.Array, u:jax.Array, w:jax.Array) -> jax.Array :
        """Get the value of the i-th component of the RHS of the dynamical system.

        Parameters
        ----------
        i : int
            component
        t : Union[Integer, Float]
            state value
        x : jax.Array
            state value
        u : jax.Array
            control value
        w : jax.Array
            disturbance value
 
        Returns
        -------
        jax.Array
            :math:`f_i(t,x,u,w)`

        """
        x = jnp.asarray(x); u = jnp.asarray(u); w = jnp.asarray(w)
        return jnp.asarray(self._fi[i](**self._txuw_to_kwargs(t,x,u,w)))
