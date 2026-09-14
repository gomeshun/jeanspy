"""Axisymmetric second moments following Hayashi & Chiba (2015), eqs. 1–5.

Independent NumPy forward API. Units: pc, Msun, km/s; inclinations in radians.
No mean streaming is specified: second moments equal dispersions only for
nonrotating systems. See docs/axisymmetric.md for assumptions and convergence.
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import lru_cache

import numpy as np
from scipy.constants import parsec
from scipy.special import roots_legendre

from ._axisymmetric_params import InvalidAxisymmetricModelError, force_limit
from ._zhao import enclosed_mass as _zhao_mass
from ._axisymmetric_components import component_models

G = 1.32712440018e20 / parsec * 1e-6
__all__ = ["AxisymmetricJeans", "PlummerTracer", "ZhaoHalo", "intrinsic_axis_ratio",
           "InvalidAxisymmetricModelError", "AxisymmetricStellarModel",
           "AxisymmetricPlummerModel", "AxisymmetricDMModel", "AxisymmetricZhaoModel",
           "AxisymmetricAnisotropyModel", "AxisymmetricConstantAnisotropyModel"]


def _positive(name, value):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")


def _inclination(value):
    if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= np.pi / 2:
        raise ValueError("inclination must be in [0, pi/2] radians")


def _coordinates(R, z):
    R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
    if R.size == 0 or np.any(~np.isfinite(R)) or np.any(R < 0) or np.any(~np.isfinite(z)):
        raise ValueError("R must be nonnegative and coordinates finite and nonempty")
    return R, z


def _rule(n):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 16:
        raise ValueError("quadrature orders must be integers >= 16")
    return _cached_rule(int(n))


@lru_cache(maxsize=16)
def _cached_rule(n):
    x, w = roots_legendre(n)
    return (x + 1) / 2, w / 2


def intrinsic_axis_ratio(q_projected, inclination):
    r"""Deproject an oblate tracer; face-on photometry is degenerate and rejected.

    Notes
    -----
    **Inputs and units.** ``q_projected`` in (0,1] and inclination in (0,pi/2]
    radians, both scalars.

    **Returns and shape.** Scalar ``q = sqrt((q_projected**2-cos(i)**2)/sin(i)**2)``.

    **Validity.** Requires ``q_projected > cos(i)``. Face-on photometry is
    degenerate and rejected.

    **Errors.** Incompatible flattening, face-on angle or invalid values raise
    ValueError.

    **Backend.** NumPy CPU.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``intrinsic_axis_ratio(0.8, np.pi/2)`` gives 0.8.
    """
    _positive("q_projected", q_projected)
    _inclination(inclination)
    if q_projected > 1 or inclination == 0:
        raise ValueError("require q_projected <= 1 and nonzero inclination")
    q2 = 1-(1-q_projected)*(1+q_projected)/np.sin(inclination)**2
    if q2 <= 0:
        raise ValueError("projected flattening is incompatible with inclination")
    return np.sqrt(q2)


class AxisymmetricStellarModel(ABC):
    """Interface for a normalized axisymmetric stellar tracer.

    Intrinsic cylindrical R>=0 and signed z, and projected x/y, are in pc.
    Concrete backends define whether parameters are stored or passed explicitly.
    """

    @abstractmethod
    def density_3d(self, R_pc, z_pc, **kwargs):
        """Return the unit-normalized tracer density in pc^-3."""

    @abstractmethod
    def surface_density(self, x_pc, y_pc, **kwargs):
        """Return the projected unit-normalized density in pc^-2."""

    @abstractmethod
    def radial_derivative(self, R_pc, z_pc, **kwargs):
        """Return the radial derivative of tracer density in pc^-4."""


class AxisymmetricDMModel(ABC):
    """Interface for a spheroidal dark-matter density and gravitational field.

    Densities are in Msun/pc^3, masses in Msun and coordinates in pc.
    """

    @abstractmethod
    def mass_density_3d(self, R_pc, z_pc, **kwargs):
        """Return the dark-matter density in Msun/pc^3."""

    @abstractmethod
    def enclosed_mass(self, m_pc, **kwargs):
        """Return mass in Msun inside a similar ellipsoid of radius m_pc."""

    @abstractmethod
    def potential_gradient(self, R_pc, z_pc, **kwargs):
        """Return (dPhi/dR,dPhi/dz) in (km/s)^2/pc."""


class AxisymmetricAnisotropyModel(ABC):
    """Interface for meridional anisotropy, beta_z = 1 - <vz²>/<vR²>.

    This is a cylindrical anisotropy, distinct from spherical beta_ani.
    The present Jeans solver supports the constant subclass only.
    """

    @abstractmethod
    def beta(self, R_pc, z_pc, **kwargs):
        """Return dimensionless cylindrical anisotropy at broadcast coordinates."""


@dataclass(frozen=True)
class AxisymmetricPlummerModel(AxisymmetricStellarModel):
    r"""Unit-integral spheroidal Plummer tracer (re_pc is equatorial scale).

    Notes
    -----
    **Inputs and units.** ``re_pc`` is equatorial scale in pc, q>0 is intrinsic
    axis ratio. Intrinsic R>=0 and signed z, or signed sky x/y, are finite and
    broadcast to one shape (pc).

    **Returns and shape.** density(R,z) is pc^-3; ``radial_derivative`` is
    dnu/dR in pc^-4; ``surface_density(x, y, inclination)`` is pc^-2;
    ``projected_axis_ratio`` returns sqrt(cos(i)^2+q^2 sin(i)^2). Array outputs
    follow the broadcast shape, including scalar output.

    **Validity.** x is the line of nodes; i=0 is face-on. No streaming
    prescription, stellar self-gravity, PSF or pixel average is supplied. All
    fixed rules require refinement.

    **Errors.** Invalid values/shapes/orders raise ValueError; wrong component
    types raise TypeError. Negative/nonfinite intrinsic moments raise
    InvalidAxisymmetricModelError. Force evaluation exactly at a cusped origin
    is unsupported.

    **Backend.** NumPy/SciPy CPU, frozen dataclasses.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_axisymmetric.py``
    """
    re_pc: float
    q: float = 1.0

    def __post_init__(self):
        _positive("re_pc", self.re_pc)
        _positive("q", self.q)

    def density(self, R, z):
        """Return the unit-normalized tracer density in pc^-3.

        Cylindrical coordinates R>=0 and signed z are in pc and broadcast.
        Nonfinite, empty or negative-R inputs raise ValueError.
        """
        R, z = _coordinates(R, z)
        return 3 / (4 * np.pi * self.q * self.re_pc**3) * (
            1 + (R**2 + (z / self.q)**2) / self.re_pc**2
        )**(-2.5)

    def radial_derivative(self, R, z):
        """Return d(nu)/dR in pc^-4 at cylindrical coordinates R,z in pc.

        Coordinates broadcast; R must be nonnegative and both arrays finite and
        nonempty. Invalid coordinates raise ValueError through density.
        """
        return -5 * np.asarray(R) * self.density(R, z) / (
            self.re_pc**2 + np.asarray(R)**2 + (np.asarray(z) / self.q)**2
        )

    def projected_axis_ratio(self, inclination):
        """Return the dimensionless projected minor-to-major tracer axis ratio.

        Inclination is a scalar in radians; zero is face-on. It must be finite
        and in [0, pi/2], otherwise ValueError is raised.
        """
        _inclination(inclination)
        return np.sqrt(np.cos(inclination)**2 + self.q**2 * np.sin(inclination)**2)

    def surface_density(self, x, y, inclination):
        """Project the unit-normalized tracer onto signed sky coordinates.

        ``x`` and ``y`` are finite, nonempty, broadcastable coordinates in pc;
        inclination is a scalar angle in radians within the class domain.
        Returns pc^-2 with the broadcast shape. Invalid coordinates or
        inclination raise ValueError.
        """
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        if x.size == 0 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            raise ValueError("sky coordinates must be finite and nonempty")
        qp = self.projected_axis_ratio(inclination)
        return (1 + (x*x + (y/qp)**2) / self.re_pc**2)**(-2) / (
            np.pi * self.re_pc**2 * qp
        )


    @property
    def a_pc(self):
        """Equatorial scale in pc (compatibility spelling of re_pc)."""
        return self.re_pc

    def density_3d(self, R_pc, z_pc):
        """Return unit-normalized tracer density in pc^-3 at coordinates in pc."""
        return self.density(R_pc, z_pc)


@dataclass(frozen=True)
class AxisymmetricZhaoModel(AxisymmetricDMModel):
    r"""rho=rhos_Msunpc3 (m/rs_pc)^(-gamma) [1+(m/rs_pc)^alpha]^((gamma-beta)/alpha).

    m²=R²+z²/Q². Q may be oblate or prolate. rhos_Msunpc3 is a density scale,
    not the density at rs_pc. Finite central potential: 0 <= gamma < 2;
    finite outer potential: beta > 2. Total mass may diverge (e.g. NFW).
    Hayashi 2015 eq. 4 is alpha=2, beta=3, gamma=-alpha_paper.

    Notes
    -----
    **Inputs and units.** ``rhos_Msunpc3`` (Msun/pc^3), ``rs_pc`` (pc), Q>0, alpha>0,
    beta>2, 0<=gamma<2; ``r_t_pc > 0`` is an ellipsoidal cutoff and can be
    infinite for the forward model. Intrinsic R>=0 and signed z, or signed sky
    x/y, are finite and broadcast to one shape (pc).

    **Returns and shape.** density is Msun/pc^3 and zero outside ``r_t_pc``.
    ``potential_gradient`` returns (dPhi/dR,dPhi/dz) in (km/s)^2/pc, opposite
    gravitational acceleration. ``enclosed_mass(m_pc)`` is Msun inside the
    ellipsoid ``R^2 + z^2/Q^2 <= m_pc^2``. J/D are scalar factors. Array outputs
    follow the broadcast shape, including scalar output.

    **Validity.** x is the line of nodes; i=0 is face-on. No streaming
    prescription, stellar self-gravity, PSF or pixel average is supplied. All
    fixed rules require refinement.

    **Errors.** Invalid values/shapes/orders raise ValueError; wrong component
    types raise TypeError. Negative/nonfinite intrinsic moments raise
    InvalidAxisymmetricModelError. Force evaluation exactly at a cusped origin
    is unsupported.

    **Backend.** NumPy/SciPy CPU, frozen dataclasses.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_axisymmetric.py``
    """
    rs_pc: float
    rhos_Msunpc3: float
    Q: float = 1.0
    alpha: float = 1.0
    beta: float = 3.0
    gamma: float = 1.0
    r_t_pc: float = np.inf

    def __post_init__(self):
        for name in ("rhos_Msunpc3", "rs_pc", "Q", "alpha"):
            _positive(name, getattr(self, name))
        if not np.isfinite(self.beta) or self.beta <= 2:
            raise ValueError("beta must be finite and > 2")
        if not np.isfinite(self.gamma) or not 0 <= self.gamma < 2:
            raise ValueError("gamma must be in [0, 2)")
        if not np.isscalar(self.r_t_pc) or np.isnan(self.r_t_pc) or self.r_t_pc <= 0:
            raise ValueError("r_t_pc must be positive (infinity is allowed)")

    def _density_slope(self, m):
        with np.errstate(divide="ignore", invalid="ignore"):
            logx = np.log(m / self.rs_pc)
            transition = np.logaddexp(0, self.alpha * logx)
            inner = np.zeros_like(logx) if self.gamma == 0 else -self.gamma * logx
            rho = self.rhos_Msunpc3 * np.exp(inner + (self.gamma-self.beta)/self.alpha * transition)
            fraction = np.exp(-np.logaddexp(0, -self.alpha * logx))
        return rho, -self.gamma + (self.gamma-self.beta) * fraction

    def density(self, R, z):
        """Evaluate halo density in Msun/pc^3 at cylindrical coordinates R,z in pc.

        Coordinates broadcast and must be finite/nonempty with R>=0, otherwise
        ValueError is raised. Density is zero beyond the ellipsoidal cutoff;
        a positive central cusp diverges at the origin.
        """
        R, z = _coordinates(R, z)
        m = np.hypot(R, z / self.Q)
        return np.where(m <= self.r_t_pc, self._density_slope(m)[0], 0.)

    def enclosed_mass(self, m_pc, *, n_steps=128):
        """Mass inside the spheroid m <= m_pc, truncated at r_t_pc, in Msun."""
        radius = np.asarray(m_pc, dtype=float)
        if (radius.size == 0 or np.any(np.isnan(radius)) or np.any(radius < 0)
                or np.any(~np.isfinite(np.minimum(radius, self.r_t_pc)))):
            raise ValueError("Require nonnegative m_pc with finite min(m_pc, r_t_pc)")
        p = dict(rs_pc=self.rs_pc, rhos_Msunpc3=self.rhos_Msunpc3, a=self.alpha,
                 b=self.beta, g=self.gamma, r_t_pc=self.r_t_pc)
        return self.Q * _zhao_mass(radius, p, xp=np, n_steps=n_steps)

    def _gradients(self, R, z, n):
        R, z = _coordinates(R, z)
        # Avoid evaluating the singular density at a cusp origin.
        if self.gamma > 0 and np.any((R == 0) & (z == 0)):
            raise ValueError("force evaluation at the exact cusp origin is not supported")
        t, w = _rule(n)
        limit, derivative = force_limit(R, z, self.Q, self.r_t_pc, np)
        t, w = limit[..., None]*t, limit[..., None]*w
        D = np.sqrt(1 + (self.Q*self.Q - 1)*t*t)
        RR, zz = R[..., None], z[..., None]
        m = t * np.sqrt(RR*RR + (zz/D)**2)
        rho, slope = self._density_slope(m)
        common = 4 * np.pi * G * self.Q * w * rho * t*t
        gR = R * np.sum(common / D, axis=-1)
        gz = z * np.sum(common / D**3, axis=-1)
        # d(gz)/dR analytically, avoiding differences of integrated pressures.
        ratio = np.divide(RR*t*t, m*m, out=np.zeros_like(m), where=m > 0)
        dgz = z * np.sum(common / D**3 * slope * ratio, axis=-1)
        if np.isfinite(self.r_t_pc):
            rho_edge = self._density_slope(self.r_t_pc)[0]
            D_edge = np.sqrt(1+(self.Q*self.Q-1)*limit*limit)
            dgz += 4*np.pi*G*self.Q*z*rho_edge*limit**2/D_edge**3*derivative
        return gR, gz, dgz

    def potential_gradient(self, R, z, n=96):
        """Return (dPhi/dR, dPhi/dz), opposite to gravitational acceleration."""
        return self._gradients(R, z, n)[:2]

    def jfactor(self, dist_pc, roi_deg, *, inclination=np.pi/2, **quadrature):
        """Finite-cone annihilation factor in GeV^2 cm^-5; finite r_t_pc required."""
        from .axisymmetric_factors import jfactor
        return jfactor(self, dist_pc, roi_deg, inclination=inclination, **quadrature)

    def dfactor(self, dist_pc, roi_deg, *, inclination=np.pi/2, **quadrature):
        """Finite-cone decay factor in GeV cm^-2; finite r_t_pc required."""
        from .axisymmetric_factors import dfactor
        return dfactor(self, dist_pc, roi_deg, inclination=inclination, **quadrature)


    @property
    def rho_s(self):
        """Density scale in Msun/pc^3 (compatibility spelling)."""
        return self.rhos_Msunpc3

    @property
    def r_s(self):
        """Scale radius in pc (compatibility spelling)."""
        return self.rs_pc

    def mass_density_3d(self, R_pc, z_pc):
        """Return halo density in Msun/pc^3, zero outside the cutoff."""
        return self.density(R_pc, z_pc)


class PlummerTracer(AxisymmetricPlummerModel):
    """Legacy constructor using a_pc in pc; prefer AxisymmetricPlummerModel.

    re_pc is accepted for dataclasses.replace; explicit a_pc takes precedence.
    """

    def __init__(self, a_pc=None, q=1., *, re_pc=None):
        super().__init__(re_pc=re_pc if a_pc is None else a_pc, q=q)


class ZhaoHalo(AxisymmetricZhaoModel):
    """Legacy constructor using rho_s and r_s; prefer AxisymmetricZhaoModel.

    Canonical parameter names are accepted for dataclasses.replace; explicit
    legacy arguments take precedence over the inherited field values.
    """

    def __init__(self, rho_s=None, r_s=None, Q=1., alpha=1., beta=3., gamma=1.,
                 r_t_pc=np.inf, *, rs_pc=None, rhos_Msunpc3=None):
        super().__init__(rs_pc=rs_pc if r_s is None else r_s,
                         rhos_Msunpc3=rhos_Msunpc3 if rho_s is None else rho_s,
                         Q=Q, alpha=alpha,
                         beta=beta, gamma=gamma, r_t_pc=r_t_pc)


@dataclass(frozen=True)
class AxisymmetricConstantAnisotropyModel(AxisymmetricAnisotropyModel):
    """Constant cylindrical anisotropy with finite beta_z < 1.

    NumPy coordinates in pc broadcast; invalid coordinates raise ValueError.
    Positive Jeans moments impose additional restrictions on the full model.
    """
    beta_z: float = 0.

    def __post_init__(self):
        if not np.isscalar(self.beta_z) or not np.isfinite(self.beta_z) or self.beta_z >= 1:
            raise ValueError("beta_z must be finite and < 1")

    def beta(self, R_pc, z_pc):
        """Return constant beta_z with the broadcast coordinate shape."""
        R, z = _coordinates(R_pc, z_pc)
        return np.full(R.shape, self.beta_z, dtype=float)


@dataclass(frozen=True)
class AxisymmetricJeans:
    r"""Aligned constant-beta_z Jeans solver with integration to infinity.

    Fixed Gauss-Legendre orders must be checked for convergence for each
    parameter regime. Large arrays are processed one sky position at a time.
    Negative intrinsic second moments raise ValueError, never get clipped.

    Notes
    -----
    **Inputs and units.** tracer is PlummerTracer; halo is ZhaoHalo;
    ``beta_z < 1``; inclination in [0,pi/2] radians; ``n_force``, ``n_vertical``
    and ``n_los`` are integer quadrature orders >=16. Intrinsic R>=0 and signed
    z, or signed sky x/y, are finite and broadcast to one shape (pc).

    **Returns and shape.** ``intrinsic_moments(R,z)`` returns (vR2,vz2,vphi2),
    each in (km/s)^2. ``los_second_moment(x,y)`` returns the
    surface-density-weighted second moment, not its square root. Array outputs
    follow the broadcast shape, including scalar output.

    **Validity.** x is the line of nodes; i=0 is face-on. No streaming
    prescription, stellar self-gravity, PSF or pixel average is supplied. All
    fixed rules require refinement.

    **Errors.** Invalid values/shapes/orders raise ValueError; wrong component
    types raise TypeError. Negative/nonfinite intrinsic moments raise
    InvalidAxisymmetricModelError. Force evaluation exactly at a cusped origin
    is unsupported.

    **Backend.** NumPy/SciPy CPU, frozen dataclasses.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_axisymmetric.py``
    """
    tracer: AxisymmetricPlummerModel
    halo: AxisymmetricZhaoModel
    beta_z: float = 0.0
    inclination: float = np.pi / 2
    n_force: int = 96
    n_vertical: int = 96
    n_los: int = 96
    anisotropy: AxisymmetricConstantAnisotropyModel | None = None

    def __post_init__(self):
        if not isinstance(self.tracer, AxisymmetricPlummerModel) or not isinstance(self.halo, AxisymmetricZhaoModel):
            raise TypeError("require AxisymmetricPlummerModel and AxisymmetricZhaoModel components")
        if not np.isscalar(self.beta_z) or not np.isfinite(self.beta_z) or self.beta_z >= 1:
            raise ValueError("beta_z must be finite and < 1")
        if self.anisotropy is not None:
            if not isinstance(self.anisotropy, AxisymmetricConstantAnisotropyModel):
                raise TypeError("Only AxisymmetricConstantAnisotropyModel is supported")
            if self.beta_z != 0. and self.beta_z != self.anisotropy.beta_z:
                raise ValueError("beta_z conflicts with the anisotropy component")
        _inclination(self.inclination)
        for n in (self.n_force, self.n_vertical, self.n_los):
            _rule(n)

    def _pressure(self, R, z):
        R, z = _coordinates(R, z)
        z = np.abs(z)
        t, w = _rule(self.n_vertical)
        scale = np.sqrt(self.tracer.a_pc**2 + R*R + (z/self.tracer.q)**2) * self.tracer.q
        zz = z[..., None] + scale[..., None] * t / (1-t)
        RR = R[..., None]
        nu = self.tracer.density(RR, zz)
        _, gz, dgz = self.halo._gradients(RR, zz, self.n_force)
        weight = scale[..., None] * w / (1-t)**2
        P = np.sum(weight * nu * gz, axis=-1)
        dP = np.sum(weight * (self.tracer.radial_derivative(RR, zz)*gz + nu*dgz), axis=-1)
        return P, dP

    def intrinsic_moments(self, R, z):
        """Return (<vR²>, <vz²>, <vphi²>) in (km/s)²; arrays broadcast."""
        R, z = _coordinates(R, z)
        P, dP = self._pressure(R, z)
        nu = self.tracer.density(R, z)
        # R*dPhi/dR vanishes on the axis, even for a central cusp.
        safe_z = np.where((R == 0) & (z == 0), self.tracer.a_pc, z)
        gR = self.halo.potential_gradient(R, safe_z, self.n_force)[0]
        vz2 = P / nu
        beta_z = (self.beta_z if self.anisotropy is None
                  else self.anisotropy.beta(R, z))
        vR2 = vz2 / (1-beta_z)
        vphi2 = (P + R*dP) / ((1-beta_z)*nu) + R*gR
        moments = np.stack([vR2, vz2, vphi2])
        if np.any(~np.isfinite(moments)) or np.any(moments < 0):
            raise InvalidAxisymmetricModelError(
                "nonfinite or negative intrinsic second moment; check model and convergence")
        return vR2, vz2, vphi2

    def los_second_moment(self, x, y):
        """Surface-density-weighted <vlos²> at sky coordinates (pc).

        x is the line of nodes. y_intrinsic=y*cos(i)+l*sin(i),
        z_intrinsic=-y*sin(i)+l*cos(i); l is the line-of-sight coordinate.
        """
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        surface = self.tracer.surface_density(x, y, self.inclination)
        t, w = _rule(self.n_los)
        si, ci = np.sin(self.inclination), np.cos(self.inclination)
        result = np.empty(x.shape)
        A = si*si + (ci/self.tracer.q)**2
        for index in np.ndindex(x.shape):
            xx, yy = x[index], y[index]
            # Center the quadrature at the maximum LOS tracer density.
            center = -yy*si*ci*(1-1/self.tracer.q**2)/A
            qp = self.tracer.projected_axis_ratio(self.inclination)
            scale = np.sqrt(self.tracer.a_pc**2 + xx*xx + (yy/qp)**2)/np.sqrt(A)
            offset = scale*t/(1-t)
            ell = center + np.concatenate([-offset, offset])
            weight = np.tile(scale*w/(1-t)**2, 2)
            Y, Z = yy*ci + ell*si, -yy*si + ell*ci
            R = np.hypot(xx, Y)
            vr, vz, vp = self.intrinsic_moments(R, Z)
            cos2 = np.divide(xx*xx, R*R, out=np.zeros_like(R), where=R > 0)
            local = si*si*((1-cos2)*vr + cos2*vp) + ci*ci*vz
            result[index] = np.sum(weight*self.tracer.density(R, Z)*local)/surface[index]
        return result[()] if result.ndim == 0 else result


@dataclass(frozen=True)
class AxisymmetricDSphModel:
    r"""Compose a NumPy tracer, halo and anisotropy into an axisymmetric model.

    Supply submodels with StellarModel=AxisymmetricPlummerModel,
    DMModel=AxisymmetricZhaoModel and
    AnisotropyModel=AxisymmetricConstantAnisotropyModel. Stored parameters are
    used when params is omitted. An explicit params mapping overrides those
    values for one call without changing the components. Without submodels,
    each call requires the full physical parameter dictionary as before.
    inclination is in radians and applies to stored components.

    Required: re_pc, rs_pc, rhos_Msunpc3 and exactly one of q/q_projected.
    Optional: Q, alpha, beta, gamma, beta_z, inclination (radians).

    Notes
    -----
    **Inputs and units.** params requires ``re_pc``, ``rs_pc`` (pc),
    ``rhos_Msunpc3`` (Msun/pc^3), and exactly one of q or ``q_projected``.
    Optional Q, alpha, beta, gamma, ``beta_z`` and inclination (radians) have
    the defaults shown in the axisymmetric guide. ``r_t_pc`` is a positive
    ellipsoidal cutoff (pc). Use alpha/beta/gamma; spherical a/b/g names are not
    accepted. Physical parameter dictionaries hold scalar values; radius arrays
    are broadcast independently. Use vmap to batch parameter dictionaries.
    Constructor node counts ``n_force``/``n_vertical``/``n_los`` are static
    integers >=16.

    **Returns and shape.** sigmalos2 and ``intrinsic_moments`` return (km/s)^2;
    ``density_3d`` is normalized pc^-3, ``surface_density`` pc^-2,
    ``mass_density_3d`` Msun/pc^3, ``enclosed_mass`` Msun inside an ellipsoid;
    ``potential_gradient`` is (km/s)^2/pc. Coordinates broadcast; intrinsic
    moments and forces are tuples of matching arrays.

    **Validity.** Cylindrical alignment with constant ``beta_z``; same physical
    restrictions as AxisymmetricJeans. Scalar sky inputs yield scalars; centers
    are allowed for projected moments.

    **Errors.** Invalid schema/values raise ValueError; nonphysical moments
    raise InvalidAxisymmetricModelError.

    **Backend.** NumPy/SciPy CPU.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_axisymmetric.py``; ``examples/docs_jax.py``
    """
    n_force: int = 96
    n_vertical: int = 96
    n_los: int = 96
    submodels: Mapping | None = None
    inclination: float = np.pi / 2

    def __post_init__(self):
        _inclination(self.inclination)
        if self.submodels is not None:
            object.__setattr__(self, "submodels", component_models(self.submodels, (
                AxisymmetricPlummerModel, AxisymmetricZhaoModel,
                AxisymmetricConstantAnisotropyModel)))
        for n in (self.n_force, self.n_vertical, self.n_los):
            _rule(n)

    def sampling_identity(self):
        """Return the three fixed quadrature orders used to identify a sampling target.

        The host dictionary contains quadrature orders, stored components and
        inclination. The likelihood separately identifies observations and priors.
        """
        return dict(n_force=self.n_force, n_vertical=self.n_vertical, n_los=self.n_los,
                    submodels=self.submodels, inclination=self.inclination)

    def __getitem__(self, name):
        if self.submodels is None:
            raise KeyError("This model has no stored components; pass explicit submodels")
        return self.submodels[name]

    @property
    def physical_params(self):
        """Detached physical defaults from stored components, or an empty mapping."""
        if self.submodels is None:
            return {}
        tracer, halo, anisotropy = (self.submodels[k] for k in (
            "StellarModel", "DMModel", "AnisotropyModel"))
        return dict(re_pc=tracer.re_pc, q=tracer.q, rs_pc=halo.rs_pc,
                    rhos_Msunpc3=halo.rhos_Msunpc3, Q=halo.Q, alpha=halo.alpha,
                    beta=halo.beta, gamma=halo.gamma, r_t_pc=halo.r_t_pc,
                    beta_z=anisotropy.beta_z, inclination=self.inclination)

    def _model(self, params):
        from ._axisymmetric_params import resolve_params
        base = self.physical_params
        stellar_type, halo_type, anisotropy_type = (
            AxisymmetricPlummerModel, AxisymmetricZhaoModel,
            AxisymmetricConstantAnisotropyModel)
        if self.submodels is not None:
            stellar_type, halo_type, anisotropy_type = (
                type(self.submodels[k]) for k in ("StellarModel", "DMModel", "AnisotropyModel"))
        if params is not None:
            if "q_projected" in params and "q" not in params:
                base.pop("q", None)
            base.update(params)
        p, valid = resolve_params(base, np)
        if not valid:
            raise InvalidAxisymmetricModelError(
                "Invalid axisymmetric physical parameters or inclination/flattening")
        p = {k: float(v) for k, v in p.items()}
        return AxisymmetricJeans(
            stellar_type(re_pc=p["re_pc"], q=p["q"]),
            halo_type(rs_pc=p["rs_pc"], rhos_Msunpc3=p["rhos_Msunpc3"], Q=p["Q"],
                      alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"], r_t_pc=p["r_t_pc"]),
            inclination=p["inclination"], n_force=self.n_force,
            n_vertical=self.n_vertical, n_los=self.n_los,
            anisotropy=anisotropy_type(beta_z=p["beta_z"]),
        )

    def sigmalos2(self, x_pc, y_pc, *, params=None):
        r"""Project a cylindrically aligned second moment.

        Notes
        -----
        **Inputs and units.** Signed ``x_pc``/``y_pc`` in pc, broadcastable
        scalar/arrays; params is the explicit physical dictionary.

        **Returns and shape.** LOS second moment in (km/s)^2 with the broadcast
        coordinate shape, including scalar output.
        """
        return self._model(params).los_second_moment(x_pc, y_pc)

    def intrinsic_moments(self, R_pc, z_pc, *, params=None):
        r"""Evaluate the intrinsic Jeans second moments.

        Notes
        -----
        **Inputs and units.** ``R_pc >= 0`` and signed ``z_pc`` in pc, broadcastable;
        params supplies the physical dictionary.

        **Returns and shape.** Tuple (vR2,vz2,vphi2), each in (km/s)^2 with the
        broadcast coordinate shape. vphi2 is the total azimuthal second moment; no
        rotation/dispersion split is assigned.
        """
        return self._model(params).intrinsic_moments(R_pc, z_pc)

    def potential_gradient(self, R_pc, z_pc, *, params=None):
        r"""Evaluate derivatives of the gravitational potential.

        Notes
        -----
        **Inputs and units.** ``R_pc >= 0``, signed ``z_pc`` in pc and explicit
        params.

        **Returns and shape.** Tuple (dPhi/dR,dPhi/dz) in (km/s)^2/pc; gravitational
        acceleration has the opposite sign.
        """
        return self._model(params).halo.potential_gradient(R_pc, z_pc, self.n_force)

    def surface_density(self, x_pc, y_pc, *, params=None):
        r"""Evaluate the projected spheroidal Plummer tracer.

        Notes
        -----
        **Inputs and units.** Signed ``x_pc``/``y_pc`` in pc and explicit params;
        coordinates broadcast.

        **Returns and shape.** Normalized surface density in pc^-2 with the
        broadcast coordinate shape.
        """
        m = self._model(params)
        return m.tracer.surface_density(x_pc, y_pc, m.inclination)

    def density_3d(self, R_pc, z_pc, *, params=None):
        """Unit-normalized stellar density in pc^-3."""
        return self._model(params).tracer.density(R_pc, z_pc)

    def mass_density_3d(self, R_pc, z_pc, *, params=None):
        """Halo density in Msun pc^-3, including the optional ellipsoidal cutoff."""
        return self._model(params).halo.density(R_pc, z_pc)

    def enclosed_mass(self, m_pc, *, params=None, n_steps=128):
        r"""Integrate mass inside a similar halo ellipsoid.

        Notes
        -----
        **Inputs and units.** ``m_pc >= 0`` is the ellipsoidal radius in pc; params
        supplies halo scales, slopes, Q and cutoff.

        **Returns and shape.** Msun inside R^2+z^2/Q^2<=``min(m_pc, r_t_pc)``^2,
        matching ``m_pc`` shape.
        """
        return self._model(params).halo.enclosed_mass(m_pc, n_steps=n_steps)

    def jfactor(self, dist_pc, roi_deg, *, params=None, **quadrature):
        r"""Postprocess an axisymmetric finite-cone factor.

        Notes
        -----
        **Inputs and units.** Scalar ``dist_pc`` and ``roi_deg``; params specifies
        halo, inclination and finite ``r_t_pc``; ``n_mu``/``n_phi``/``n_radial`` set
        independent factor quadratures, all >=16.

        **Returns and shape.** Python float in GeV^2 cm^-5.

        **Validity.** Require explicit finite ``r_t_pc``, observer distance >
        ``r_t_pc``\*max(1,Q), ``0 <= roi_deg < 90`` and gamma<1.5 (finite central
        annihilation integral). ``n_phi`` uses a periodic rule; refine all orders.
        """
        model = self._model(params)
        return model.halo.jfactor(dist_pc, roi_deg, inclination=model.inclination, **quadrature)

    def dfactor(self, dist_pc, roi_deg, *, params=None, **quadrature):
        r"""Postprocess an axisymmetric finite-cone factor.

        Notes
        -----
        **Inputs and units.** Scalar ``dist_pc`` and ``roi_deg``; params specifies
        halo, inclination and finite ``r_t_pc``; ``n_mu``/``n_phi``/``n_radial`` set
        independent factor quadratures, all >=16.

        **Returns and shape.** Python float in GeV cm^-2.

        **Validity.** Require explicit finite ``r_t_pc``, observer distance >
        ``r_t_pc``\*max(1,Q), ``0 <= roi_deg < 90`` and gamma<2 under the ZhaoHalo
        constructor domain. ``n_phi`` uses a periodic rule; refine all orders.
        """
        model = self._model(params)
        return model.halo.dfactor(dist_pc, roi_deg, inclination=model.inclination, **quadrature)


__all__.append("AxisymmetricDSphModel")
