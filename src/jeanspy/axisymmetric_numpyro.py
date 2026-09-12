"""Differentiable axisymmetric Jeans forward API for JAX/NumPyro.

The numerical core uses only JAX operations (no SciPy callbacks). Configure
JEANSPY_JAX_ENABLE_X64=true before importing for scientific double precision.
Invalid physical proposals produce NaN moments and are rejected by the likelihood.
"""
from dataclasses import dataclass
from functools import partial

from ._jax_env import configure_jax_environment
configure_jax_environment()

import jax
import jax.numpy as jnp

from .axisymmetric import G, _rule
from ._axisymmetric_params import resolve_params

__all__ = ["AxisymmetricDSphModel"]


def _coords(a, b):
    a, b = jnp.broadcast_arrays(jnp.asarray(a, dtype=float), jnp.asarray(b, dtype=float))
    if not a.size:
        raise ValueError("Coordinates must be nonempty")
    valid = jnp.isfinite(a) & jnp.isfinite(b)
    return jnp.where(valid, a, 0.), jnp.where(valid, b, 0.), valid


def _sqrt_nonnegative(x):
    # sqrt(0) has an infinite derivative even when its tangent vanishes.
    return jnp.where(x > 0, jnp.sqrt(jnp.where(x > 0, x, 1.)), 0.)


@dataclass(frozen=True)
class AxisymmetricDSphModel:
    """Same parameters and units as axisymmetric.AxisymmetricDSphModel.

    Fixed quadrature settings are static under JIT. lax.map with rematerialized
    per-star evaluation bounds intermediate memory when differentiating catalogs.
    """
    n_force: int = 96
    n_vertical: int = 96
    n_los: int = 96

    def __post_init__(self):
        for n in (self.n_force, self.n_vertical, self.n_los):
            _rule(n)

    def sampling_identity(self):
        return dict(n_force=self.n_force, n_vertical=self.n_vertical, n_los=self.n_los)

    def _force(self, R, z, p):
        t, w = (jnp.asarray(v) for v in _rule(self.n_force))
        D = jnp.sqrt(1+(p["Q"]**2-1)*t*t)
        RR, zz = R[..., None], z[..., None]
        m2 = t*t*(RR*RR+(zz/D)**2)
        # Avoid log(0) at the core center. Force components vanish there.
        m = jnp.sqrt(jnp.where(m2 > 0, m2, p["rs_pc"]**2))
        logx = jnp.log(m/p["rs_pc"])
        logtransition = jnp.logaddexp(0., p["alpha"]*logx)
        rho = p["rhos_Msunpc3"]*jnp.exp(-p["gamma"]*logx+
                (p["gamma"]-p["beta"])/p["alpha"]*logtransition)
        slope = -p["gamma"]+(p["gamma"]-p["beta"])*jax.nn.sigmoid(p["alpha"]*logx)
        common = 4*jnp.pi*G*p["Q"]*w*rho*t*t
        gR = R*jnp.sum(common/D, axis=-1)
        gz = z*jnp.sum(common/D**3, axis=-1)
        ratio = RR*t*t/jnp.where(m2 > 0, m2, 1.)
        dgz = z*jnp.sum(common/D**3*slope*ratio, axis=-1)
        return gR, gz, dgz

    def _nu(self, R, z, p):
        # Normalization cancels; omitting a^-3/q improves float32 dynamic range.
        return (1+(R/p["re_pc"])**2+(z/(p["q"]*p["re_pc"]))**2)**(-2.5)

    def _intrinsic(self, R, z, p):
        t, w = (jnp.asarray(v) for v in _rule(self.n_vertical))
        zabs = jnp.abs(z)
        scale = jnp.sqrt(p["re_pc"]**2+R*R+(zabs/p["q"])**2)*p["q"]
        zz = zabs[..., None]+scale[..., None]*t/(1-t)
        RR = R[..., None]
        nu = self._nu(RR, zz, p)
        _, gz, dgz = self._force(RR, zz, p)
        dnu = -5*RR*nu/(p["re_pc"]**2+RR*RR+(zz/p["q"])**2)
        weight = scale[..., None]*w/(1-t)**2
        P = jnp.sum(weight*nu*gz, axis=-1)
        dP = jnp.sum(weight*(dnu*gz+nu*dgz), axis=-1)
        vz2 = P/self._nu(R, z, p)
        vr2 = vz2/(1-p["beta_z"])
        gr = self._force(R, jnp.where((R == 0)&(z == 0), p["re_pc"], z), p)[0]
        vp2 = (P+R*dP)/((1-p["beta_z"])*self._nu(R,z,p))+R*gr
        return jnp.stack([vr2, vz2, vp2])

    @partial(jax.jit, static_argnums=0)
    def intrinsic_moments(self, R_pc, z_pc, *, params):
        p, valid = resolve_params(params, jnp)
        R, z, coords_valid = _coords(R_pc, z_pc)
        coords_valid = coords_valid & (R >= 0)
        moments = self._intrinsic(jnp.maximum(R, 0.), z, p)
        physical = jnp.all(jnp.isfinite(moments)&(moments >= 0), axis=0)
        return tuple(jnp.where(valid & coords_valid & physical, v, jnp.nan) for v in moments)

    @partial(jax.jit, static_argnums=0)
    def potential_gradient(self, R_pc, z_pc, *, params):
        p, valid = resolve_params(params, jnp)
        R, z, coords_valid = _coords(R_pc, z_pc)
        valid = valid & coords_valid & (R >= 0) & ~((R == 0)&(z == 0)&(p["gamma"] > 0))
        return tuple(jnp.where(valid, g, jnp.nan) for g in self._force(jnp.maximum(R,0.),z,p)[:2])

    @partial(jax.jit, static_argnums=0)
    def surface_density(self, x_pc, y_pc, *, params):
        p, valid = resolve_params(params, jnp)
        x,y,coords_valid = _coords(x_pc,y_pc)
        qp = jnp.sqrt(jnp.cos(p["inclination"])**2+p["q"]**2*jnp.sin(p["inclination"])**2)
        surface = (1+(x*x+(y/qp)**2)/p["re_pc"]**2)**(-2)/(jnp.pi*p["re_pc"]**2*qp)
        return jnp.where(valid & coords_valid, surface, jnp.nan)

    @partial(jax.jit, static_argnums=0)
    def sigmalos2(self, x_pc, y_pc, *, params):
        p, valid = resolve_params(params, jnp)
        x,y,coords_valid = _coords(x_pc,y_pc)
        t,w = (jnp.asarray(v) for v in _rule(self.n_los))
        si,ci = jnp.sin(p["inclination"]),jnp.cos(p["inclination"])
        A = si*si+(ci/p["q"])**2
        qp = jnp.sqrt(ci*ci+p["q"]**2*si*si)

        def one(xy):
            xx,yy = xy
            center = -yy*si*ci*(1-1/p["q"]**2)/A
            scale = jnp.sqrt(p["re_pc"]**2+xx*xx+(yy/qp)**2)/jnp.sqrt(A)
            offset = scale*t/(1-t)
            ell = center+jnp.concatenate([-offset,offset])
            weight = jnp.tile(scale*w/(1-t)**2,2)
            Y,Z = yy*ci+ell*si,-yy*si+ell*ci
            R2 = xx*xx+Y*Y
            R = _sqrt_nonnegative(R2)
            moments = self._intrinsic(R,Z,p)
            physical = jnp.all(jnp.isfinite(moments)&(moments >= 0))
            vr,vz,vp = moments
            cos2 = xx*xx/jnp.where(R2 > 0,R2,1.)
            local = si*si*((1-cos2)*vr+cos2*vp)+ci*ci*vz
            # Surface density with the same unnormalized nu as _intrinsic.
            surface = 4*p["q"]*p["re_pc"]/(3*qp)*(1+(xx*xx+(yy/qp)**2)/p["re_pc"]**2)**(-2)
            value = jnp.sum(weight*self._nu(R,Z,p)*local)/surface
            return jnp.where(physical,value,jnp.nan)

        result = jax.lax.map(jax.checkpoint(one), jnp.stack([x.ravel(),y.ravel()],axis=-1)).reshape(x.shape)
        return jnp.where(valid & coords_valid,result,jnp.nan)
