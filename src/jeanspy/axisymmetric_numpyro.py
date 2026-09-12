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
from ._axisymmetric_params import force_limit, resolve_params
from ._zhao import enclosed_mass as _zhao_mass

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
    r"""Same parameters and units as axisymmetric.AxisymmetricDSphModel.

    Fixed quadrature settings are static under JIT. lax.map with rematerialized
    per-star evaluation bounds intermediate memory when differentiating catalogs.

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

    **Errors.** Invalid dynamic parameters yield NaN, including under jit.
    Schema/shape/configuration errors raise before evaluation; likelihood
    rejects invalid variances.

    **Backend.** JAX arrays on the configured CPU/GPU, with dtype set before
    import.

    **Differentiation.** Physical scalar parameters and supported coordinate
    values are differentiable in admissible smooth regions. Node counts, schema
    decisions, rejection masks and hard-cutoff boundaries are not continuous
    model parameters. J/D methods are not part of this JAX class.

    **Examples.** ``examples/docs_axisymmetric.py``; ``examples/docs_jax.py``
    """
    n_force: int = 96
    n_vertical: int = 96
    n_los: int = 96

    def __post_init__(self):
        for n in (self.n_force, self.n_vertical, self.n_los):
            _rule(n)

    def sampling_identity(self):
        """Return the three fixed quadrature orders used to identify a sampling target.

        The host dictionary contains n_force, n_vertical and n_los. This metadata
        helper has no physical-parameter derivative.
        """
        return dict(n_force=self.n_force, n_vertical=self.n_vertical, n_los=self.n_los)

    def _force(self, R, z, p):
        t, w = (jnp.asarray(v) for v in _rule(self.n_force))
        limit, derivative = force_limit(R, z, p["Q"], p["r_t_pc"], jnp)
        t, w = limit[..., None]*t, limit[..., None]*w
        D = jnp.sqrt(1+(p["Q"]**2-1)*t*t)
        RR, zz = R[..., None], z[..., None]
        m2 = t*t*(RR*RR+(zz/D)**2)
        # Avoid log(0) at the core center. Force components vanish there.
        m = jnp.sqrt(jnp.where(m2 > 0, m2, p["rs_pc"]**2))
        logx = jnp.log(m/p["rs_pc"])
        logtransition = jnp.logaddexp(0., p["alpha"]*logx)
        rho = p["rhos_Msunpc3"]*jnp.exp(-p["gamma"]*logx+
                (p["gamma"]-p["beta"])/p["alpha"]*logtransition)
        # At a cored origin the force vanishes, but its coordinate derivative
        # is proportional to the central density, not the benign log placeholder.
        rho = jnp.where(m2 > 0, rho, p["rhos_Msunpc3"])
        slope = -p["gamma"]+(p["gamma"]-p["beta"])*jax.nn.sigmoid(p["alpha"]*logx)
        common = 4*jnp.pi*G*p["Q"]*w*rho*t*t
        gR = R*jnp.sum(common/D, axis=-1)
        gz = z*jnp.sum(common/D**3, axis=-1)
        ratio = RR*t*t/jnp.where(m2 > 0, m2, 1.)
        dgz = z*jnp.sum(common/D**3*slope*ratio, axis=-1)
        rt = jnp.where(jnp.isfinite(p["r_t_pc"]), p["r_t_pc"], p["rs_pc"])
        edge_logx = jnp.log(rt/p["rs_pc"])
        edge_density = p["rhos_Msunpc3"]*jnp.exp(-p["gamma"]*edge_logx +
            (p["gamma"]-p["beta"])/p["alpha"]*jnp.logaddexp(0., p["alpha"]*edge_logx))
        edge_D = jnp.sqrt(1+(p["Q"]**2-1)*limit**2)
        dgz += 4*jnp.pi*G*p["Q"]*z*edge_density*limit**2/edge_D**3*derivative
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
        r"""Evaluate the intrinsic Jeans second moments.

        Notes
        -----
        **Inputs and units.** ``R_pc``>=0 and signed ``z_pc`` in pc, broadcastable;
        params supplies the physical dictionary.

        **Returns and shape.** Tuple (vR2,vz2,vphi2), each in (km/s)^2 with the
        broadcast coordinate shape. vphi2 is the total azimuthal second moment; no
        rotation/dispersion split is assigned.
        """
        p, valid = resolve_params(params, jnp)
        R, z, coords_valid = _coords(R_pc, z_pc)
        coords_valid = coords_valid & (R >= 0)
        moments = self._intrinsic(jnp.where(R >= 0, R, 0.), z, p)
        physical = jnp.all(jnp.isfinite(moments)&(moments >= 0), axis=0)
        return tuple(jnp.where(valid & coords_valid & physical, v, jnp.nan) for v in moments)

    @partial(jax.jit, static_argnums=0)
    def potential_gradient(self, R_pc, z_pc, *, params):
        r"""Evaluate derivatives of the gravitational potential.

        Notes
        -----
        **Inputs and units.** ``R_pc``>=0, signed ``z_pc`` in pc and explicit
        params.

        **Returns and shape.** Tuple (dPhi/dR,dPhi/dz) in (km/s)^2/pc; gravitational
        acceleration has the opposite sign.
        """
        p, valid = resolve_params(params, jnp)
        R, z, coords_valid = _coords(R_pc, z_pc)
        valid = valid & coords_valid & (R >= 0) & ~((R == 0)&(z == 0)&(p["gamma"] > 0))
        return tuple(jnp.where(valid, g, jnp.nan) for g in self._force(jnp.where(R >= 0, R, 0.),z,p)[:2])

    @partial(jax.jit, static_argnums=0)
    def surface_density(self, x_pc, y_pc, *, params):
        r"""Evaluate the projected spheroidal Plummer tracer.

        Notes
        -----
        **Inputs and units.** Signed ``x_pc``/``y_pc`` in pc and explicit params;
        coordinates broadcast.

        **Returns and shape.** Normalized surface density in pc^-2 with the
        broadcast coordinate shape.
        """
        p, valid = resolve_params(params, jnp)
        x,y,coords_valid = _coords(x_pc,y_pc)
        qp = jnp.sqrt(jnp.cos(p["inclination"])**2+p["q"]**2*jnp.sin(p["inclination"])**2)
        surface = (1+(x*x+(y/qp)**2)/p["re_pc"]**2)**(-2)/(jnp.pi*p["re_pc"]**2*qp)
        return jnp.where(valid & coords_valid, surface, jnp.nan)

    @partial(jax.jit, static_argnums=0)
    def density_3d(self, R_pc, z_pc, *, params):
        """Unit-normalized stellar density in pc^-3."""
        p, valid = resolve_params(params, jnp)
        R, z, coords_valid = _coords(R_pc, z_pc)
        value = self._nu(R, z, p)*3/(4*jnp.pi*p["q"]*p["re_pc"]**3)
        return jnp.where(valid & coords_valid & (R >= 0), value, jnp.nan)

    @partial(jax.jit, static_argnums=0)
    def mass_density_3d(self, R_pc, z_pc, *, params):
        """Halo density, including the optional ellipsoidal cutoff."""
        p, valid = resolve_params(params, jnp)
        R, z, coords_valid = _coords(R_pc, z_pc)
        m = _sqrt_nonnegative(R*R+(z/p["Q"])**2)
        logx = jnp.log(jnp.where(m > 0, m/p["rs_pc"], 1.))
        rho = p["rhos_Msunpc3"]*jnp.exp(-p["gamma"]*logx +
                  (p["gamma"]-p["beta"])/p["alpha"]*jnp.logaddexp(0., p["alpha"]*logx))
        rho = jnp.where(m == 0, jnp.where(p["gamma"] == 0, p["rhos_Msunpc3"], jnp.inf), rho)
        rho = jnp.where(m <= p["r_t_pc"], rho, 0.)
        return jnp.where(valid & coords_valid & (R >= 0), rho, jnp.nan)

    @partial(jax.jit, static_argnums=0, static_argnames=("n_steps",))
    def enclosed_mass(self, m_pc, *, params, n_steps=128):
        r"""Mass in Msun inside the spheroid m <= m_pc (with truncation).

        Notes
        -----
        **Inputs and units.** ``m_pc``>=0 is the ellipsoidal radius in pc; params
        supplies halo scales, slopes, Q and cutoff.

        **Returns and shape.** Msun inside R^2+z^2/Q^2<=min(``m_pc``,``r_t_pc``)^2,
        matching ``m_pc`` shape.
        """
        p, valid = resolve_params(params, jnp)
        r = jnp.asarray(m_pc, dtype=float)
        if not r.size:
            raise ValueError("m_pc must be nonempty")
        mass_params = dict(rs_pc=p["rs_pc"], rhos_Msunpc3=p["rhos_Msunpc3"],
                           a=p["alpha"], b=p["beta"], g=p["gamma"], r_t_pc=p["r_t_pc"])
        mass = p["Q"]*_zhao_mass(r, mass_params, xp=jnp, n_steps=n_steps)
        return jnp.where(valid, mass, jnp.nan)

    @partial(jax.jit, static_argnums=0)
    def sigmalos2(self, x_pc, y_pc, *, params):
        r"""Project a cylindrically aligned second moment.

        Notes
        -----
        **Inputs and units.** Signed ``x_pc``/``y_pc`` in pc, broadcastable
        scalar/arrays; params is the explicit physical dictionary.

        **Returns and shape.** LOS second moment in (km/s)^2 with the broadcast
        coordinate shape, including scalar output.
        """
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
