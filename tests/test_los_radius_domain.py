"""Positive-radius contract and independent near-center LOS validation."""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import log_density
from scipy.integrate import quad

from jeanspy import model as classical
from jeanspy import model_numpyro as functional
from jeanspy.sampler_numpyro import JeansLikelihoodModel


PARAMS = dict(re_pc=200., rs_pc=1000., rhos_Msunpc3=.01,
              r_t_pc=10000., beta_ani=0., vmem_kms=0.)


@contextmanager
def precision(x64):
    old = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", x64)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def model(module):
    kw = (lambda *names: {k: PARAMS[k] for k in names}) if module is classical else (lambda *names: {})
    return module.DSphModel(**kw("vmem_kms"), submodels={
        "StellarModel": module.PlummerModel(**kw("re_pc")),
        "DMModel": module.NFWModel(**kw("rs_pc", "rhos_Msunpc3", "r_t_pc")),
        "AnisotropyModel": module.ConstantAnisotropyModel(**kw("beta_ani")),
    })


def independent_isotropic_los(R):
    """Adaptive LOS integral in physical z, without JeansPy kernels or grids.

    Swapping isotropic Jeans and projection integrals gives
    2G/Sigma(R) integral_R^inf nu(r) M(r) sqrt(r^2-R^2)/r^2 dr.
    r=hypot(R,z) removes its endpoint root. Dimensionless z/Re is integrated
    by QUADPACK. At R=0 this also supplies the independent limiting value.
    """
    re, rs, rho, rt = (PARAMS[k] for k in ("re_pc", "rs_pc", "rhos_Msunpc3", "r_t_pc"))

    def integrand(t):
        z = re * t
        r = np.hypot(R, z)
        x = min(r, rt) / rs
        # Positive convergent NFW series near the center avoids cancellation.
        shape = (x*x*(.5+x*(-2/3+x*(.75+x*(-.8+x*5/6))))) if x < 1e-3 else np.log1p(x)-x/(1+x)
        mass = 4*np.pi*rho*rs**3*shape
        nu = 3/(4*np.pi*re**3)*(1+(r/re)**2)**(-2.5)
        return nu*mass*z*z/r**3 * re

    surface = 1/(np.pi*re**2)*(1+(R/re)**2)**(-2)
    integral = quad(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-10, limit=400)[0]
    return 2 * (1.32712440018e20 / 3.085677581491367e16) * 1e-6 / surface * integral


@pytest.mark.parametrize("radius", [0., -1., np.inf, -np.inf, np.nan])
def test_classical_rejects_invalid_scalar_and_mixed_radii(radius):
    dsph = model(classical)
    for value in (radius, [50., radius, 300.]):
        for evaluate in (dsph.sigmalos2, dsph.sigmalos2_dequad, dsph.sigmalos_dequad):
            with pytest.raises(ValueError, match="finite R_pc > 0"):
                evaluate(value)


@pytest.mark.parametrize("backend", ["kernel", "abel"])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("x64", [False, True])
def test_jax_masks_only_invalid_elements_and_preserves_shape(backend, jit, x64):
    with precision(x64):
        dsph = model(functional)
        kwargs = dict(params=PARAMS, backend=backend, jit=jit)
        valid = jnp.array([50., 300.])
        mixed = jnp.array([50., 0., -1., jnp.nan, jnp.inf, -jnp.inf, 300.])
        actual = dsph.sigmalos2(mixed, **kwargs)
        assert actual.shape == mixed.shape
        np.testing.assert_allclose(actual[jnp.array([0, 6])], dsph.sigmalos2(valid, **kwargs), rtol=1e-6)
        assert np.isnan(actual[1:6]).all()
        for scalar in (0., -1., jnp.nan, jnp.inf):
            result = dsph.sigmalos2(scalar, **kwargs)
            assert result.shape == (1,)
            assert np.isnan(result).all()
        assert np.shape(dsph.sigmalos2(50, **kwargs)) == (1,)
        direct = getattr(dsph, "sigmalos2_" + backend)
        assert np.isnan(direct(0., params=PARAMS)).all()
        likelihood = JeansLikelihoodModel(dsph, [], parameter_postprocess=lambda _: PARAMS,
                                          sigmalos2_kwargs={"backend": backend, "jit": jit})
        value, _ = log_density(likelihood, (mixed, jnp.zeros(7), jnp.ones(7)), {}, {})
        assert np.isneginf(value)


@pytest.mark.parametrize("radius", [[], [[50.]], np.ones((2, 2))])
def test_bad_input_shape_rejected(radius):
    with pytest.raises(ValueError, match="scalar or nonempty one-dimensional"):
        model(classical).sigmalos2(radius)
    for jit in (False, True):
        for backend in ("kernel", "abel"):
            with pytest.raises(ValueError, match="scalar or nonempty one-dimensional"):
                model(functional).sigmalos2(jnp.asarray(radius), params=PARAMS, jit=jit, backend=backend)


@pytest.mark.parametrize("backend", ["kernel", "abel"])
def test_near_center_against_independent_integral_and_convergence(backend):
    with precision(True):
        radii = np.array([.001, .01, .1, 1., 50., 300.])
        reference = np.array([independent_isotropic_los(r) for r in radii])
        center = independent_isotropic_los(0.)
        assert center > 20.
        assert reference[0] == pytest.approx(center, rel=1e-7)
        np.testing.assert_allclose(model(classical).sigmalos2(radii), reference, rtol=1e-6)
        assert np.shape(model(classical).sigmalos2(.001)) == ()
        dsph = model(functional)
        # Small R needs a larger u_max (r=R*u). These radii deliberately lie
        # outside the maintained R/Re>=.005 default accuracy envelope.
        kwargs = dict(backend=backend, params=PARAMS, u_max=1e8, n_u=1025, n_r=8192)
        actual = dsph.sigmalos2(jnp.asarray(radii), **kwargs)
        np.testing.assert_allclose(actual, reference, rtol=5e-4)
        refined = dsph.sigmalos2(jnp.asarray(radii), **dict(kwargs, n_u=2049, n_r=16384, u_max=2e8))
        np.testing.assert_allclose(actual, refined, rtol=5e-4)


@pytest.mark.parametrize("backend", ["kernel", "abel"])
def test_positive_radius_gradients_are_finite_and_match_differences(backend):
    with precision(True):
        dsph = model(functional)
        radii = jnp.array([50., 130., 300.])
        def prediction(r):
            return dsph.sigmalos2(r, params=PARAMS, backend=backend, n_r=1024).sum()
        gradient = np.asarray(jax.jit(jax.grad(prediction))(radii))
        assert np.isfinite(gradient).all()
        delta = 1e-4
        expected = [(float(prediction(radii+jnp.eye(3)[i]*delta))
                     - float(prediction(radii-jnp.eye(3)[i]*delta)))/(2*delta) for i in range(3)]
        np.testing.assert_allclose(gradient, expected, rtol=1e-4, atol=1e-8)


def test_classical_invalid_density_or_kernel_never_becomes_zero():
    dsph = model(classical)
    dsph["StellarModel"].density_3d = lambda r: np.full_like(r, np.nan)
    with pytest.raises(ValueError, match="Stellar density"):
        dsph.sigmalos2([50., 100.])
    dsph = model(classical)
    dsph["AnisotropyModel"].kernel = lambda u, R, **kw: np.full((len(R), u.size), np.nan)
    with pytest.raises(ValueError, match="Nonfinite LOS integrand"):
        dsph.sigmalos2([50., 100.])
