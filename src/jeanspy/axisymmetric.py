"""Axisymmetric second moments following Hayashi & Chiba (2015), eqs. 1–5.

Independent NumPy forward API. Units: pc, Msun, km/s; inclinations in radians.
No mean streaming is specified: second moments equal dispersions only for
nonrotating systems. See docs/axisymmetric.md for assumptions and convergence.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.constants import parsec
from scipy.special import roots_legendre

G = 1.32712440018e20 / parsec * 1e-6
__all__ = ["AxisymmetricJeans", "PlummerTracer", "ZhaoHalo", "intrinsic_axis_ratio"]


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


@lru_cache(maxsize=16)
def _rule(n):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 16:
        raise ValueError("quadrature orders must be integers >= 16")
    x, w = roots_legendre(n)
    return (x + 1) / 2, w / 2


def intrinsic_axis_ratio(q_projected, inclination):
    """Deproject an oblate tracer; face-on photometry is degenerate and rejected."""
    _positive("q_projected", q_projected)
    _inclination(inclination)
    if q_projected > 1 or inclination == 0:
        raise ValueError("require q_projected <= 1 and nonzero inclination")
    q2 = (q_projected**2 - np.cos(inclination)**2) / np.sin(inclination)**2
    if q2 <= 0:
        raise ValueError("projected flattening is incompatible with inclination")
    return np.sqrt(q2)


@dataclass(frozen=True)
class PlummerTracer:
    """Unit-integral spheroidal Plummer tracer (a_pc is equatorial scale)."""
    a_pc: float
    q: float = 1.0

    def __post_init__(self):
        _positive("a_pc", self.a_pc)
        _positive("q", self.q)

    def density(self, R, z):
        R, z = _coordinates(R, z)
        return 3 / (4 * np.pi * self.q * self.a_pc**3) * (
            1 + (R**2 + (z / self.q)**2) / self.a_pc**2
        )**(-2.5)

    def radial_derivative(self, R, z):
        return -5 * np.asarray(R) * self.density(R, z) / (
            self.a_pc**2 + np.asarray(R)**2 + (np.asarray(z) / self.q)**2
        )

    def projected_axis_ratio(self, inclination):
        _inclination(inclination)
        return np.sqrt(np.cos(inclination)**2 + self.q**2 * np.sin(inclination)**2)

    def surface_density(self, x, y, inclination):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        if x.size == 0 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            raise ValueError("sky coordinates must be finite and nonempty")
        qp = self.projected_axis_ratio(inclination)
        return (1 + (x*x + (y/qp)**2) / self.a_pc**2)**(-2) / (
            np.pi * self.a_pc**2 * qp
        )


@dataclass(frozen=True)
class ZhaoHalo:
    """rho=rho_s (m/r_s)^(-gamma) [1+(m/r_s)^alpha]^((gamma-beta)/alpha).

    m²=R²+z²/Q². Q may be oblate or prolate. rho_s is a density scale,
    not the density at r_s. Finite central potential: 0 <= gamma < 2;
    finite outer potential: beta > 2. Total mass may diverge (e.g. NFW).
    Hayashi 2015 eq. 4 is alpha=2, beta=3, gamma=-alpha_paper.
    """
    rho_s: float
    r_s: float
    Q: float = 1.0
    alpha: float = 1.0
    beta: float = 3.0
    gamma: float = 1.0

    def __post_init__(self):
        for name in ("rho_s", "r_s", "Q", "alpha"):
            _positive(name, getattr(self, name))
        if not np.isfinite(self.beta) or self.beta <= 2:
            raise ValueError("beta must be finite and > 2")
        if not np.isfinite(self.gamma) or not 0 <= self.gamma < 2:
            raise ValueError("gamma must be in [0, 2)")

    def _density_slope(self, m):
        with np.errstate(divide="ignore", invalid="ignore"):
            logx = np.log(m / self.r_s)
            transition = np.logaddexp(0, self.alpha * logx)
            inner = np.zeros_like(logx) if self.gamma == 0 else -self.gamma * logx
            rho = self.rho_s * np.exp(inner + (self.gamma-self.beta)/self.alpha * transition)
            fraction = np.exp(-np.logaddexp(0, -self.alpha * logx))
        return rho, -self.gamma + (self.gamma-self.beta) * fraction

    def density(self, R, z):
        R, z = _coordinates(R, z)
        return self._density_slope(np.hypot(R, z / self.Q))[0]

    def _gradients(self, R, z, n):
        R, z = _coordinates(R, z)
        # Avoid evaluating the singular density at a cusp origin.
        if self.gamma > 0 and np.any((R == 0) & (z == 0)):
            raise ValueError("force evaluation at the exact cusp origin is not supported")
        t, w = _rule(n)
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
        return gR, gz, dgz

    def potential_gradient(self, R, z, n=96):
        """Return (dPhi/dR, dPhi/dz), opposite to gravitational acceleration."""
        return self._gradients(R, z, n)[:2]


@dataclass(frozen=True)
class AxisymmetricJeans:
    """Aligned constant-beta_z Jeans solver with integration to infinity.

    Fixed Gauss-Legendre orders must be checked for convergence for each
    parameter regime. Large arrays are processed one sky position at a time.
    Negative intrinsic second moments raise ValueError, never get clipped.
    """
    tracer: PlummerTracer
    halo: ZhaoHalo
    beta_z: float = 0.0
    inclination: float = np.pi / 2
    n_force: int = 96
    n_vertical: int = 96
    n_los: int = 96

    def __post_init__(self):
        if not isinstance(self.tracer, PlummerTracer) or not isinstance(self.halo, ZhaoHalo):
            raise TypeError("require PlummerTracer and ZhaoHalo components")
        if not np.isscalar(self.beta_z) or not np.isfinite(self.beta_z) or self.beta_z >= 1:
            raise ValueError("beta_z must be finite and < 1")
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
        vR2 = vz2 / (1-self.beta_z)
        vphi2 = (P + R*dP) / ((1-self.beta_z)*nu) + R*gR
        moments = np.stack([vR2, vz2, vphi2])
        if np.any(~np.isfinite(moments)) or np.any(moments < 0):
            raise ValueError("nonfinite or negative intrinsic second moment; check model and convergence")
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
    """Parameter-dictionary forward API shared with the JAX backend.

    Required: re_pc, rs_pc, rhos_Msunpc3 and exactly one of q/q_projected.
    Optional: Q, alpha, beta, gamma, beta_z, inclination (radians).
    """
    n_force: int = 96
    n_vertical: int = 96
    n_los: int = 96

    def __post_init__(self):
        for n in (self.n_force, self.n_vertical, self.n_los):
            _rule(n)

    def sampling_identity(self):
        return dict(n_force=self.n_force, n_vertical=self.n_vertical, n_los=self.n_los)

    def _model(self, params):
        from ._axisymmetric_params import resolve_params
        p, valid = resolve_params(params, np)
        if not valid:
            raise ValueError("Invalid axisymmetric physical parameters or inclination/flattening")
        p = {k: float(v) for k, v in p.items()}
        return AxisymmetricJeans(
            PlummerTracer(p["re_pc"], p["q"]),
            ZhaoHalo(p["rhos_Msunpc3"], p["rs_pc"], p["Q"], p["alpha"], p["beta"], p["gamma"]),
            p["beta_z"], p["inclination"], self.n_force, self.n_vertical, self.n_los,
        )

    def sigmalos2(self, x_pc, y_pc, *, params):
        return self._model(params).los_second_moment(x_pc, y_pc)

    def intrinsic_moments(self, R_pc, z_pc, *, params):
        return self._model(params).intrinsic_moments(R_pc, z_pc)

    def potential_gradient(self, R_pc, z_pc, *, params):
        return self._model(params).halo.potential_gradient(R_pc, z_pc, self.n_force)

    def surface_density(self, x_pc, y_pc, *, params):
        m = self._model(params)
        return m.tracer.surface_density(x_pc, y_pc, m.inclination)


__all__.append("AxisymmetricDSphModel")
