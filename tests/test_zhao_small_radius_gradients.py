"""Independent closed-form regressions for the weak core scale derivative."""
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jeanspy import model as classical
from jeanspy import model_jax
from jeanspy.axisymmetric_jax import AxisymmetricZhaoModel


@contextmanager
def precision(enabled):
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("x64,rtol", [(True, 2e-12), (False, 2e-5)])
@pytest.mark.parametrize("gamma", [0., 1e-20, .5, 1.])
def test_scale_score_matches_closed_form_zhao_family(x64, rtol, gamma):
    # alpha=3-gamma, beta=6-gamma gives
    # M = 4*pi*rho_s*r_s^3/p * x^p/(1+x^p), p=3-gamma.
    # Its logarithmic rs derivative at fixed other parameters is
    # gamma + p*x^p/(1+x^p). It remains nonzero for a core at tiny x.
    with precision(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        rs = jnp.asarray(1e3, dtype=dtype)
        # At still smaller radii in float32, dM itself underflows before
        # log(M) rescales it. This change removes cancellation, not underflow.
        radii = jnp.asarray(
            ([1e-6] if x64 else []) + [1e-3, .1, 10., 999., 1000., 1001., 1e5],
            dtype=dtype,
        )
        p = 3 - gamma
        params = dict(rhos_Msunpc3=.01, alpha=p, beta=6-gamma,
                      gamma=gamma, r_t_pc=jnp.inf)
        model = model_jax.ZhaoModel()

        def objective(logrs):
            return jnp.log(model.enclosed_mass(
                radii, params=dict(params, rs_pc=jnp.exp(logrs))))

        logrs = jnp.log(rs)
        # Use the actual represented scale in the reference (float32 exp/log).
        x = np.asarray(radii, dtype=float) / float(jnp.exp(logrs))
        expected = gamma + p*x**p/(1+x**p)
        forward = jax.jit(jax.jacfwd(objective))(logrs)
        reverse = jax.jit(jax.jacrev(objective))(logrs)
        np.testing.assert_allclose(forward, expected, rtol=rtol, atol=0.)
        np.testing.assert_allclose(reverse, expected, rtol=rtol, atol=0.)
        assert np.all(np.asarray(forward) > 0)


def test_second_scale_derivative_and_truncated_radius():
    with precision(True):
        radii = jnp.array([0., .001, .1, 1., 10.])
        params = dict(rhos_Msunpc3=.01, alpha=3., beta=6., gamma=0., r_t_pc=1.)
        model = model_jax.ZhaoModel()

        def mass(logrs):
            return model.enclosed_mass(radii, params=dict(params, rs_pc=jnp.exp(logrs)))

        logrs = jnp.log(1000.)
        assert float(mass(logrs)[0]) == 0.
        assert float(jax.jacfwd(mass)(logrs)[0]) == 0.
        score = jax.jacfwd(lambda value: jnp.log(mass(value)[1:]))
        second = jax.jit(jax.jacfwd(score))(logrs)
        t = (np.minimum(np.asarray(radii[1:]), 1.) / float(jnp.exp(logrs)))**3
        np.testing.assert_allclose(second, -9*t/(1+t)**2, rtol=2e-12, atol=0.)


def test_axisymmetric_mass_inherits_stable_scale_derivative():
    with precision(True):
        model = AxisymmetricZhaoModel()
        params = dict(rhos_Msunpc3=.01, alpha=3., beta=6., gamma=0.,
                      r_t_pc=jnp.inf, Q=.7)

        def objective(logrs):
            return jnp.log(model.enclosed_mass(
                .001, params=dict(params, rs_pc=jnp.exp(logrs))))

        logrs = jnp.log(1000.)
        x = .001 / float(jnp.exp(logrs))
        np.testing.assert_allclose(jax.jit(jax.grad(objective))(logrs),
                                   3*x**3/(1+x**3), rtol=2e-12, atol=0.)


@pytest.mark.parametrize("x64,radius,rtol", [(True, 1e-120, 2e-12), (False, 1e-16, 2e-5)])
def test_combined_prefactor_does_not_underflow_for_steep_cusp(x64, radius, rtol):
    # r**3 * (r/rs)**(-gamma) would evaluate as 0*inf although M is finite.
    with precision(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        params = {key: jnp.asarray(value, dtype=dtype) for key, value in dict(
            rs_pc=1e3, rhos_Msunpc3=.01, alpha=.1, beta=3.1,
            gamma=2.9, r_t_pc=jnp.inf).items()}
        # Use alpha=p and beta=3+p at the precision under test.
        params["alpha"] = 3 - params["gamma"]
        params["beta"] = 3 + params["alpha"]
        p = float(params["alpha"])
        x = radius / float(params["rs_pc"])
        expected = 4*np.pi*float(params["rhos_Msunpc3"])*float(params["rs_pc"])**3/p * x**p/(1+x**p)
        evaluate = jax.jit(lambda r: model_jax.ZhaoModel().enclosed_mass(r, params=params))
        np.testing.assert_allclose(evaluate(jnp.asarray(radius, dtype=dtype)),
                                   expected, rtol=rtol, atol=0.)
        if x64:
            actual = classical.ZhaoModel(**{k: float(v) for k, v in params.items()}).enclosed_mass(radius)
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0.)
