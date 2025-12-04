import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, Callable, Sequence, Iterable
import numpy as np
from functools import partial
from itertools import accumulate, product
from itertools import permutations as perms
from jax._src.api import api_boundary
from .interval import Interval, interval
from .nif import natif


def jacM(f: Callable[..., jax.Array]) -> Callable[..., Interval]:
    """Creates the M matrices for the Jacobian-based inclusion function.

    All positional arguments are assumed to be replaced with interval arguments for the inclusion function.

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Jacobian Inclusion Function from

    Returns
    -------
    Callable[..., Interval]
        Jacobian-Based Inclusion Function of f

    """

    @jit
    @api_boundary
    def F(
        *args, centers: jax.Array | Sequence[jax.Array] | None = None, **kwargs
    ) -> Interval:
        """Jacobian-based Inclusion Function of f.

        All positional arguments from f should be replaced with interval arguments for the inclusion function.

        Additional Args:
            centers (jax.Array | Sequence[jax.Array] | None, optional): _description_. Defaults to None.

        Parameters
        ----------
        *args :

        centers:jax.Array|Sequence[jax.Array]|None :
             (Default value = None)
        **kwargs :


        Returns
        -------
        Interval
            Interval output from the Jacobian-based Inclusion Function

        """
        args = [interval(arg).atleast_1d() for arg in args]
        if centers is None:
            centers = [tuple([(x.lower + x.upper) / 2 for x in args])]
        elif isinstance(centers, jax.Array):
            centers = [centers]
        elif not isinstance(centers, Sequence):
            raise Exception(
                "Must pass jax.Array (one center), Sequence[jax.Array], or None (auto-centered) for the centers argument"
            )

        # return [natif(jax.jacfwd(partial(f, **kwargs), i))(*args) for i in range(len(args))]
        return [
            natif(jax.jacrev(partial(f, **kwargs), i))(*args) for i in range(len(args))
        ]
        # return [interval(jax.jacfwd(f, i)(*centers[0])) for i in range(len(args))]

    return F


def jacif(f: Callable[..., jax.Array]) -> Callable[..., Interval]:
    """Creates a Jacobian Inclusion Function of f using natif.

    All positional arguments are assumed to be replaced with interval arguments for the inclusion function.

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Jacobian Inclusion Function from

    Returns
    -------
    Callable[..., Interval]
        Jacobian-Based Inclusion Function of f

    """

    @jit
    @api_boundary
    def F(
        *args, centers: jax.Array | Sequence[jax.Array] | None = None, **kwargs
    ) -> Interval:
        """Jacobian-based Inclusion Function of f.

        All positional arguments from f should be replaced with interval arguments for the inclusion function.

        Additional Args:
            centers (jax.Array | Sequence[jax.Array] | None, optional): _description_. Defaults to None.

        Parameters
        ----------
        *args :

        centers:jax.Array|Sequence[jax.Array]|None :
             (Default value = None)
        **kwargs :


        Returns
        -------
        Interval
            Interval output from the Jacobian-based Inclusion Function

        """
        args = [interval(arg).atleast_1d() for arg in args]
        if centers is None:
            centers = [tuple([(x.lower + x.upper) / 2 for x in args])]
        elif isinstance(centers, jax.Array):
            centers = [centers]
        elif not isinstance(centers, Sequence):
            raise Exception(
                "Must pass jax.Array (one center), Sequence[jax.Array], or None (auto-centered) for the centers argument"
            )

        retl, retu = [], []
        df = [natif(jax.jacrev(f, i))(*args) for i in range(len(args))]
        # df = [natif(jax.jacfwd(f, i))(*args) for i in range(len(args))]
        for center in centers:
            if len(center) != len(args):
                raise Exception(
                    f"Not enough points {len(center)=} != {len(args)=} to center the Jacobian-based inclusion function around"
                )
            f0 = f(*center)
            sum = interval(f0)
            for i in range(len(args)):
                # sum = natif(jnp.add)(sum, natif(jnp.matmul)(df[i], (interval(args[i].lower - center[i], args[i].upper - center[i]))))
                sum = sum + interval(df[i]) @ (
                    interval(args[i].lower - center[i], args[i].upper - center[i])
                )
            retl.append(sum.lower)
            retu.append(sum.upper)
        retl, retu = jnp.array(retl), jnp.array(retu)
        return interval(jnp.max(retl, axis=0), jnp.min(retu, axis=0))

    return F


class Permutation(tuple):
    """A tuple of :math:`n` numbers :math:`(o_i)_{i=1}^n` such that each :math:`0\\leq o_i \\leq n-1` and each :math:`o_i` is unique."""

    def __new__(cls, __iterable: Iterable = ()) -> "Permutation":
        if sum([2**x for x in __iterable]) != (2 ** len(__iterable) - 1):
            raise Exception(
                f"The permutation doesnt have every i from 0 to n-1: {__iterable}"
            )
        return tuple.__new__(Permutation, (__iterable))

    def __str__(self) -> str:
        return "Permutation" + super().__str__()

    def sub(self, i: int) -> "Permutation":
        """Returns the sub-permutation of the first i elements."""
        return self[: i + 1]

    @property
    def arr(self) -> jax.Array:
        """Returns the Permutation in a jax.Array."""
        return jnp.asarray(self)

    @property
    def mat(self) -> jax.Array:
        """Returns the permutation matrix of the Permutation."""
        n = len(self)
        return jnp.array(
            [[1 if j == self[i] else 0 for j in range(n)] for i in range(n)]
        )

    @property
    def mtx(self) -> jax.Array:
        """Returns the replacement matrix of the Permutation."""
        n = len(self)
        return jnp.array(
            [[1 if j in self.sub(i) else 0 for j in range(n)] for i in range(n)]
        )


def standard_permutation(n: int) -> Tuple[Permutation]:
    """Returns the standard n-permutation :math:`(0,\\dots,n-1)`"""
    return (Permutation(range(n)),)


def two_permutations(n: int) -> Tuple[Permutation]:
    """Returns the two standard n-permutations :math:`(0,\\dots,n-1)` and :math:`(n-1,\\dots,0)`"""
    return (Permutation(range(n)), Permutation(tuple(reversed(range(n)))))


def all_permutations(n: int) -> Tuple[Permutation]:
    """Returns all n-permutations."""
    return tuple(Permutation(x) for x in perms(range(n)))


class Corner(tuple):
    """A tuple of :math:`n` elements in :math:`\\{0,1\\}` representing the corners of an :math:`n`-dimensional hypercube. 0 is the lower bound, 1 is the upper bound"""

    def __new__(cls, __iterable: Iterable = ()) -> "Corner":
        for x in __iterable:
            if x not in (0, 1):
                raise Exception(f"The corner elements need to be in 0,1: {__iterable}")
        return tuple.__new__(Corner, (__iterable))

    def __str__(self) -> str:
        return "Corner" + super().__str__()


def bot_corner(n: int) -> Tuple[Corner]:
    """Returns the bottom corner of the n-dimensional hypercube."""
    return (Corner((0,) * n),)


def top_corner(n: int) -> Tuple[Corner]:
    """Returns the top corner of the n-dimensional hypercube."""
    return (Corner((1,) * n),)


def two_corners(n: int) -> Tuple[Corner]:
    """Returns the bottom and top corners of the n-dimensional hypercube."""
    return (Corner((0,) * n), Corner((1,) * n))


def all_corners(n: int) -> Tuple[Corner]:
    """Returns all corners of the n-dimensional hypercube."""
    return tuple(Corner(x) for x in product((0, 1), repeat=n))


def get_corner(M: Interval, c: Corner):
    """Gets the corner of the interval M specified by the Corner c."""
    sh = M.shape
    M = M.reshape(-1)
    return jnp.array(
        [M.lower[i] if ci == 0 else M.upper[i] for i, ci in enumerate(c)]
    ).reshape(sh)


def get_corners(M: Interval, cs: Tuple[Corner] | None = None):
    """Gets the corners of the interval M specified by the Corners cs. Defaults to all corners if None."""
    if cs is None:
        cs = all_corners(M.size)
    return [get_corner(M, c) for c in cs]
    # return [(Mc := get_corner(M, c)) for c in cs if not jnp.allclose(Mc, 0)]


def get_sparse_corners(M: Interval):
    """Gets the corners of the interval M that are not the same. Will"""
    sh = M.shape
    M = M.reshape(-1)
    ic = jnp.isclose(M.lower, M.upper)
    cs = [
        Corner(p) for p in product(*[(0,) if ic[i] else (0, 1) for i in range(len(ic))])
    ]
    return [
        jnp.array(
            [M.lower[i] if ci == 0 else M.upper[i] for i, ci in enumerate(c)]
        ).reshape(sh)
        for c in cs
    ]


def mjacM(
    f: Callable[..., jax.Array],
    argnums: int | Sequence[int] | None = None,
    has_aux: bool = False,
    mode: str = "rev",
) -> Callable:
    """Creates the M matrices for the Mixed Jacobian-based inclusion function.

    All positional arguments are assumed to be replaced with interval arguments for the inclusion function.

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Mixed Jacobian Inclusion Function from
    argnums : int | Sequence[int] | None, optional
        Integer or sequence of integers specifying which positional arguments to
        differentiate with respect to. If None, differentiates with respect to all
        positional arguments. Aligns with JAX's jacfwd/jacrev interface.
    has_aux : bool, optional
        If True, f returns a tuple (output, aux) where aux is auxiliary data not
        to be differentiated. The transform will return (matrices, aux). Default is False.
    mode : {"fwd", "rev", "auto"}, optional
        Differentiation mode: "fwd" for jacfwd, "rev" for jacrev, "auto" to choose
        based on input/output dimensions. Default is "rev".

    Returns
    -------
    Callable
        Transform that accepts:
        - *args: Interval arguments
        - permutation: Single Permutation (not tuple)
        - center: Single center point (not tuple)
        - corner: Optional single Corner (not tuple)
        - **kwargs: Additional kwargs to f

        Returns: List of M matrices (one per argnum), or (matrices, aux) if has_aux

    """

    @api_boundary
    def F(
        *args,
        permutation: Permutation | None = None,
        center: jax.Array | Sequence[jax.Array] | None = None,
        corner: Corner | None = None,
        # Old-style parameters that should error
        permutations: Tuple[Permutation] | None = None,
        centers: jax.Array | Sequence[jax.Array] | None = None,
        corners: Tuple[Corner] | None = None,
        **kwargs,
    ) -> list:
        """Mixed Jacobian M-matrix computation with single permutation and center.

        Parameters
        ----------
        *args : Interval arguments
        permutation : Permutation | None, optional
            Single permutation to use. If None, uses standard permutation.
        center : jax.Array | Sequence[jax.Array] | None, optional
            Single center point. If None, auto-centers at midpoint.
        corner : Corner | None, optional
            Single corner to use instead of center.
        permutations : Tuple[Permutation] | None, optional
            OLD INTERFACE - Will raise error if used
        centers : jax.Array | Sequence[jax.Array] | None, optional
            OLD INTERFACE - Will raise error if used
        corners : Tuple[Corner] | None, optional
            OLD INTERFACE - Will raise error if used

        Returns
        -------
        list
            List of M matrices (one per argnum), or (matrices, aux) if has_aux

        Raises
        ------
        ValueError
            If old-style interface is used (permutations, centers, corners as tuples)
        """
        # Check for old-style interface usage
        if permutations is not None:
            raise ValueError(
                "The 'permutations' parameter is no longer supported. "
                "Use 'permutation' (singular) with a single Permutation instead. "
                "For multiple permutations, use jax.vmap over the transform."
            )
        if centers is not None:
            raise ValueError(
                "The 'centers' parameter is no longer supported. "
                "Use 'center' (singular) with a single center point instead. "
                "For multiple centers, use jax.vmap over the transform."
            )
        if corners is not None:
            raise ValueError(
                "The 'corners' parameter is no longer supported. "
                "Use 'corner' (singular) with a single Corner instead. "
                "For multiple corners, use jax.vmap over the transform."
            )

        # Process arguments
        args = [interval(arg).atleast_1d() for arg in args]
        leninputsfull = tuple([len(x) for x in args])
        leninputs = sum(leninputsfull)

        # Determine which arguments to differentiate
        if argnums is None:
            _argnums = range(len(args))
        elif isinstance(argnums, int):
            _argnums = (argnums,)
        else:
            _argnums = argnums

        # Set default permutation
        if permutation is None:
            permutation = standard_permutation(leninputs)[0]

        cumsum = tuple(accumulate(leninputsfull))

        # Handle center/corner
        if center is None:
            if corner is None:
                # Auto-centered at midpoint
                center = tuple([(x.lower + x.upper) / 2 for x in args])
            else:
                # Use corner
                center = tuple(
                    [
                        (x.lower if corner[i] == 0 else x.upper)
                        for i, x in enumerate(args)
                    ]
                )
        elif isinstance(center, jax.Array):
            # Single center point - keep as is
            center = (center,) if len(args) == 1 else center
        elif not isinstance(center, Sequence):
            raise ValueError(
                "Must pass jax.Array (one center), Sequence[jax.Array], or None (auto-centered) for the center argument"
            )

        # Validate center length
        if len(center) != len(args):
            raise ValueError(
                f"Not enough points {len(center)=} != {len(args)=} to center the Jacobian-based inclusion function around"
            )

        # Helper functions for concatenation
        def arg2z(*args):
            return jnp.concatenate(args)

        def z2arg(z):
            return jnp.split(z, cumsum[:-1], axis=-1)

        # Select differentiation mode
        if mode == "rev":
            jac_func = jax.jacrev
        elif mode == "fwd":
            jac_func = jax.jacfwd
        elif mode == "auto":
            # Simple heuristic: use forward if fewer inputs than outputs
            # This is a placeholder - in practice we'd need to know output dim
            jac_func = jax.jacrev  # Default to rev for now
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'fwd', 'rev', or 'auto'.")

        # Create derivative functions for specified argnums
        if has_aux:
            # Wrap f to separate output and aux
            def f_main(*args, **kwargs):
                out, _ = f(*args, **kwargs)
                return out

            df_func = [natif(jac_func(partial(f_main, **kwargs), i)) for i in _argnums]
            # Get aux by evaluating at center
            _, aux = f(*center, **kwargs)
        else:
            df_func = [natif(jac_func(partial(f, **kwargs), i)) for i in _argnums]

        # Compute bounds
        _z = arg2z(*[arg.lower for arg in args])
        z_ = arg2z(*[arg.upper for arg in args])
        zc = arg2z(*center)

        # Build interval array for permutation
        Z = interval(
            jnp.where(
                permutation.mtx,
                jnp.tile(_z, (len(permutation), 1)),
                jnp.tile(zc, (len(permutation), 1)),
            ),
            jnp.where(
                permutation.mtx,
                jnp.tile(z_, (len(permutation), 1)),
                jnp.tile(zc, (len(permutation), 1)),
            ),
        )

        _cumsum = (0,) + cumsum
        retc = []
        npsig = np.asarray(permutation)

        # Build M matrices using vmap
        for i, argnum in enumerate(_argnums):
            idx = np.logical_and(npsig >= _cumsum[argnum], npsig < _cumsum[argnum + 1])
            Mi = jax.vmap(df_func[i])(*natif(z2arg)(Z[idx]))
            retc.append(
                Mi[
                    np.arange(leninputsfull[argnum]), :, np.arange(leninputsfull[argnum])
                ].T
            )

        if has_aux:
            return retc, aux
        else:
            return retc

    return F


def mjacif(f: Callable[..., jax.Array]) -> Callable[..., Interval]:
    """Creates a Mixed Jacobian Inclusion Function of f using natif.

    All positional arguments are assumed to be replaced with interval arguments for the inclusion function.

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Mixed Jacobian Inclusion Function from

    Returns
    -------
    Callable[..., Interval]
        Mixed Jacobian-Based Inclusion Function of f

    """

    # @wraps(f)
    @partial(jit, static_argnames=["permutations", "corners"])
    @api_boundary
    def F(
        *args,
        permutations: Tuple[Permutation] | None = None,
        centers: jax.Array | Sequence[jax.Array] | None = None,
        corners: Tuple[Corner] | None = None,
        **kwargs,
    ) -> Interval:
        """_summary_

        Parameters
        ----------
        permutations : Tuple[Permutation] | None, optional
            _description_, by default None
        centers : jax.Array | Sequence[jax.Array] | None, optional
            _description_, by default None
        corners : Tuple[Corner] | None, optional
            _description_, by default None

        Returns
        -------
        Interval
            _description_

        Raises
        ------
        Exception
            _description_
        Exception
            _description_
        Exception
            _description_
        Exception
            _description_
        Exception
            _description_
        """

        args = [interval(arg).atleast_1d() for arg in args]
        leninputsfull = tuple([len(x) for x in args])
        leninputs = sum(leninputsfull)

        if permutations is None:
            permutations = standard_permutation(leninputs)
        elif isinstance(permutations, Permutation):
            permutations = [permutations]
        elif not isinstance(permutations, Tuple):
            raise Exception(
                "Must pass jax.Array (one permutation), Sequence[jax.Array], or None (auto standard permutation) for the permutations argument"
            )

        cumsum = tuple(accumulate(leninputsfull))
        permutations_pairs = []

        # Split each permutation into individual inputs and indices.
        # Permutation is of the length of taking each input and concatenating them.
        # The result in permutations_pairs is a list of tuples of 2-tuples (argi, subi)
        # argi is the argument index, subi is the subindex of that argument.
        for permutation in permutations:
            if len(permutation) != leninputs:
                raise Exception(
                    f"The permutation is not the same length as the sum of the lengths of the inputs: {len(permutation)} != {leninputs}"
                )
            pairs = []
            for o in permutation:
                a = 0
                while cumsum[a] - 1 < o:
                    a += 1
                pairs.append((a, (o - cumsum[a - 1] if a > 0 else o)))
            permutations_pairs.append(tuple(pairs))

        # Mixed Centered
        if centers is None:
            if corners is None:
                # Auto-centered
                centers = [tuple([(x.lower + x.upper) / 2 for x in args])]
            else:
                centers = []
        elif isinstance(centers, jax.Array):
            centers = [centers]
        elif not isinstance(centers, Sequence):
            raise Exception(
                "Must pass jax.Array (one center), Sequence[jax.Array], or None (auto-centered) for the centers argument"
            )

        if corners is not None:
            if not isinstance(corners, Tuple):
                raise Exception(
                    "Must pass Tuple[Corner] or None for the corners argument"
                )
            centers.extend(
                [
                    tuple(
                        [
                            (x.lower if c[i] == 0 else x.upper)
                            for i, x in enumerate(args)
                        ]
                    )
                    for c in corners
                ]
            )

        # multiple permutations/centers, need to take min/max for final inclusion function output.
        retl, retu = [], []

        # This is the \sfJ_x for each argument. Natural inclusion on the Jacobian tree.
        # TODO: we change to jacrev here, jacfwd doesn't work for unclear reasons
        df_func = [natif(jax.jacrev(f, i)) for i in range(len(args))]

        # centers is an array of centers to check
        for center in centers:
            if len(center) != len(args):
                raise Exception(
                    f"Not enough points {len(center)=} != {len(args)=} to center the Jacobian-based inclusion function around"
                )
            f0 = f(*center)
            for pairs in permutations_pairs:
                # argr initialized at the center, will be slowly relaxed to the whole interval
                argr = [interval(c) for c in center]
                # val is the eventual output, M ([\ulx,\olx] - \overcirc{x}).
                # Will be built column-by-column of the matrix multiplication.
                val = interval(f0)

                for argi, subi in pairs:
                    # Perform the replacement operation using (argi, subi)
                    l = argr[argi].lower.at[subi].set(args[argi].lower[subi])
                    u = argr[argi].upper.at[subi].set(args[argi].upper[subi])
                    # Set the "running" interval
                    argr[argi] = interval(l, u)
                    # Mixed Jacobian matrix subi-th column. TODO: Make more efficient?
                    # Extracting a column here for the right shape for the multiplication
                    Mi = df_func[argi](*argr, **kwargs)[:, (subi,)]
                    # Mi ([\ulx,\olx]_i - \overcirc{x}_i)
                    val = natif(jnp.add)(
                        val,
                        natif(jnp.matmul)(
                            Mi,
                            (
                                interval(
                                    args[argi].lower[subi] - center[argi][subi],
                                    args[argi].upper[subi] - center[argi][subi],
                                ).reshape(-1)
                            ),
                        ),
                    )

                # (\sfJ_x, \overcirc{x}, \calO)-Mixed Jacobian-based added to the potential
                retl.append(val.lower)
                retu.append(val.upper)

        retl, retu = jnp.array(retl), jnp.array(retu)
        return interval(jnp.max(retl, axis=0), jnp.min(retu, axis=0))

    return F
