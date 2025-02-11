from ..dual_star import DualStar
import jax.numpy as jnp
from jaxtyping import ArrayLike
from ...utils import null_space
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Annulus, Patch
from matplotlib.axes import Axes
import numpy as onp
from matplotlib.path import Path
# import transforms from matplotlib
from matplotlib import transforms


class Ellipsoid (DualStar) :
    def __init__ (self, ox, H, uy) :
        super().__init__(ox, [lambda x: x.T @ x], [H], [0.], [uy])

    def V (self, x:ArrayLike) :
        P = self.H[0].T @ self.H[0]
        return (x - self.ox).T @ P @ (x - self.ox)
    
    def plot_projection (self, ax, xi=0, yi=1, rescale=False, **kwargs) :
        P = self.H[0].T @ self.H[0] / self.uy[0]
        n = P.shape[0]
        if n == 2 :
            _plot_ellipse (P, self.ox, ax, rescale, **kwargs)
            return
        ind = [k for k in range(n) if k not in [xi,yi]]
        Phat = P[ind,:]
        N = null_space(Phat)
        M = N[(xi,yi),:] # Since M is guaranteed 2x2,
        Minv = (1/(M[0,0]*M[1,1] - M[0,1]*M[1,0]))*jnp.array([[M[1,1], -M[0,1]], [-M[1,0], M[0,0]]])
        Q = Minv.T@N.T@P@N@Minv
        _plot_ellipse(Q, self.ox[(xi,yi),], ax, rescale, **kwargs)

    def __repr__(self) :
        return f'Ellipsoid(P={self.P}, xc={self.xc})'
    
    def __str__(self) :
        return f'Ellipsoid(P={self.P}, xc={self.xc})'

# def iover (e:Ellipsoid) -> irx.Interval :
#     """Interval over-approximation of an Ellipsoid"""
#     overpert = jnp.sqrt(jnp.diag(e.Pinv))
#     return irx.icentpert(e.xc, overpert)

# def eover (ix:irx.Interval, P:jax.Array) -> Ellipsoid :
#     """Ellipsoid over-approximation of an Interval"""
#     xc, xp = irx.i2centpert(ix)
#     corns = irx.get_corners(ix - xc)
#     m = jnp.max(jnp.array([norm_P(c, P) for c in corns]))
#     return Ellipsoid(P/m, xc)

class EllipsoidAnnulus (DualStar) :
    def __init__ (self, ox, H, ly, uy) :
        super().__init__(ox, [lambda x: x.T @ x], [H], [ly], [uy])

    def V (self, x:ArrayLike) :
        P = self.H[0].T @ self.H[0]
        return (x - self.ox).T @ P @ (x - self.ox)
    
    def plot_projection (self, ax, xi=0, yi=1, rescale=False, **kwargs) :
        P = self.H[0].T @ self.H[0] / self.uy[0]
        n = P.shape[0]
        if n == 2 :
            _plot_annulus (P, self.ox, self.ly[0]/self.uy[0], ax, rescale, **kwargs)
            return
        ind = [k for k in range(n) if k not in [xi,yi]]
        Phat = P[ind,:]
        N = null_space(Phat)
        M = N[(xi,yi),:] # Since M is guaranteed 2x2,
        Minv = (1/(M[0,0]*M[1,1] - M[0,1]*M[1,0]))*jnp.array([[M[1,1], -M[0,1]], [-M[1,0], M[0,0]]])
        Q = Minv.T@N.T@P@N@Minv
        _plot_annulus(Q, self.ox[(xi,yi),], self.ly[0]/self.uy[0], ax, rescale, **kwargs)

    def __repr__(self) :
        return f'EllipsoidAnnulus(P={self.P}, xc={self.xc})'
    
    def __str__(self) :
        return f'EllipsoidAnnulus(P={self.P}, xc={self.xc})'

def _plot_ellipse (Q:ArrayLike, xc:ArrayLike=jnp.zeros(2), ax:Axes|None=None, rescale:bool=False, **kwargs) :
    """
    Parameters
    ----------
    Q : ArrayLike
        PD matrix defining the ellipse
    xc : ArrayLike, optional
        Center of the ellipse, by default jnp.zeros(2)
    ax : Axes | None, optional
        Matplotlib Axes object to plot the ellipse on, plt.gca() if None, by default None
    rescale : bool, optional
        Rescales the axes to fit the ellipse, by default False

    Raises
    ------
    ValueError
        Q must be a 2x2 matrix
    """
    n = Q.shape[0]
    if n != 2 :
        raise ValueError("Use _plot_ellipse for 2D ellipses, see Ellipsoid.plot_projection")

    S, U = jnp.linalg.eigh(Q)
    Sinv = 1/S

    kwargs.setdefault('color', 'k')
    kwargs.setdefault('fill', False)
    width, height = 2*jnp.sqrt(Sinv)
    angle = jnp.arctan2(U[1, 0], U[0, 0]) * 180 / jnp.pi
    ellipse = Ellipse(xy=xc, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

    if rescale :
        ax.set_xlim(xc[0] - 1.5*width, xc[0] + 1.5*width)
        ax.set_ylim(xc[1] - 1.5*height, xc[1] + 1.5*height)


def _plot_annulus (Q:ArrayLike, xc:ArrayLike=jnp.zeros(2), inner:float=0.5, ax:Axes|None=None, rescale:bool=False, **kwargs) :
    """
    Parameters
    ----------
    Q : ArrayLike
        PD matrix defining the ellipse
    xc : ArrayLike, optional
        Center of the ellipse, by default jnp.zeros(2)
    inner: float, optional
        Inner radius of the annulus, by default 0.5
    ax : Axes | None, optional
        Matplotlib Axes object to plot the ellipse on, plt.gca() if None, by default None
    rescale : bool, optional
        Rescales the axes to fit the ellipse, by default False

    Raises
    ------
    ValueError
        Q must be a 2x2 matrix
    """
    n = Q.shape[0]
    if n != 2 :
        raise ValueError("Use _plot_ellipse for 2D ellipses, see Ellipsoid.plot_projection")

    S, U = jnp.linalg.eigh(Q)
    Sinv = 1/S

    kwargs.setdefault('color', 'k')
    kwargs.setdefault('fill', False)
    width, height = 2*jnp.sqrt(Sinv)
    angle = jnp.arctan2(U[1, 0], U[0, 0]) * 180 / jnp.pi
    # outer_ellipse = Ellipse(xy=xc, width=width, height=height, angle=angle, **kwargs)
    # inner_ellipse = Ellipse(xy=xc, width=inner*width, height=inner*height, angle=angle, **kwargs)
    annulus = _AnnulusP(xy=xc, r=jnp.sqrt(Sinv), width=inner, angle=angle, **kwargs)
    ax.add_patch(annulus)

    if rescale :
        ax.set_xlim(xc[0] - 1.5*width, xc[0] + 1.5*width)
        ax.set_ylim(xc[1] - 1.5*height, xc[1] + 1.5*height)

# def _plot_3d_ellipsoid ()


class _AnnulusP(Patch):
    """
    An elliptical annulus.
    Most of the following code is from matplotlib.patches.Annulus.
    There are small modification to make the inner ellipse defined as
        a percentage of the outer ellipse---scaling major and minor axes
        as a multiplier of the outer ellipse's major and minor axes instead of additive.
    """

    # @_docstring.interpd
    def __init__(self, xy, r, width, angle=0.0, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            xy coordinates of annulus centre.
        r : float or (float, float)
            The radius, or semi-axes:

            - If float: radius of the outer circle.
            - If two floats: semi-major and -minor axes of outer ellipse.
        width : float
            Width (thickness) of the annular ring. The width is measured inward
            from the outer ellipse so that for the inner ellipse the semi-axes
            are given by ``r - width``. *width* must be less than or equal to
            the semi-minor axis.
        angle : float, default: 0
            Rotation angle in degrees (anti-clockwise from the positive
            x-axis). Ignored for circular annuli (i.e., if *r* is a scalar).
        **kwargs
            Keyword arguments control the `Patch` properties:

            %(Patch:kwdoc)s
        """
        super().__init__(**kwargs)

        self.set_radii(r)
        self.center = xy
        self.width = width
        self.angle = angle
        self._path = None

    def __str__(self):
        if self.a == self.b:
            r = self.a
        else:
            r = (self.a, self.b)

        return "Annulus(xy=(%s, %s), r=%s, width=%s, angle=%s)" % \
                (*self.center, r, self.width, self.angle)

    def set_center(self, xy):
        """
        Set the center of the annulus.

        Parameters
        ----------
        xy : (float, float)
        """
        self._center = xy
        self._path = None
        self.stale = True

    def get_center(self):
        """Return the center of the annulus."""
        return self._center

    center = property(get_center, set_center)

    def set_width(self, width):
        """
        Set the width (thickness) of the annulus ring.

        The width is measured as a percent of both minor and major axes.

        Parameters
        ----------
        width : float
        """
        if width > 1 or width < 0:
            raise ValueError(
                'Width of annulus must be a float between 0 and 1.')

        self._width = width
        self._path = None
        self.stale = True

    def get_width(self):
        """Return the width (thickness) of the annulus ring."""
        return self._width

    width = property(get_width, set_width)

    def set_angle(self, angle):
        """
        Set the tilt angle of the annulus.

        Parameters
        ----------
        angle : float
        """
        self._angle = angle
        self._path = None
        self.stale = True

    def get_angle(self):
        """Return the angle of the annulus."""
        return self._angle

    angle = property(get_angle, set_angle)

    def set_semimajor(self, a):
        """
        Set the semi-major axis *a* of the annulus.

        Parameters
        ----------
        a : float
        """
        self.a = float(a)
        self._path = None
        self.stale = True

    def set_semiminor(self, b):
        """
        Set the semi-minor axis *b* of the annulus.

        Parameters
        ----------
        b : float
        """
        self.b = float(b)
        self._path = None
        self.stale = True

    def set_radii(self, r):
        """
        Set the semi-major (*a*) and semi-minor radii (*b*) of the annulus.

        Parameters
        ----------
        r : float or (float, float)
            The radius, or semi-axes:

            - If float: radius of the outer circle.
            - If two floats: semi-major and -minor axes of outer ellipse.
        """
        if onp.shape(r) == (2,):
            self.a, self.b = r
        elif onp.shape(r) == ():
            self.a = self.b = float(r)
        else:
            raise ValueError("Parameter 'r' must be one or two floats.")

        self._path = None
        self.stale = True

    def get_radii(self):
        """Return the semi-major and semi-minor radii of the annulus."""
        return self.a, self.b

    radii = property(get_radii, set_radii)

    def _transform_verts(self, verts, a, b):
        return transforms.Affine2D() \
            .scale(*self._convert_xy_units((a, b))) \
            .rotate_deg(self.angle) \
            .translate(*self._convert_xy_units(self.center)) \
            .transform(verts)

    def _recompute_path(self):
        # circular arc
        arc = Path.arc(0, 360)

        # annulus needs to draw an outer ring
        # followed by a reversed and scaled inner ring
        a, b, w = self.a, self.b, self.width
        v1 = self._transform_verts(arc.vertices, a, b)
        v2 = self._transform_verts(arc.vertices[::-1], a*w, b*w)
        v = onp.vstack([v1, v2, v1[0, :], (0, 0)])
        c = onp.hstack([arc.codes, Path.MOVETO,
                       arc.codes[1:], Path.MOVETO,
                       Path.CLOSEPOLY])
        self._path = Path(v, c)

    def get_path(self):
        if self._path is None:
            self._recompute_path()
        return self._path
