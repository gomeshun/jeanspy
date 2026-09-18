"""Zhao release regressions: independent mass references and inference safety."""
from contextlib import contextmanager
from functools import lru_cache
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from numpyro.infer.util import log_density

from jeanspy import model as classical
from jeanspy import model_jax as functional
from jeanspy.sampler_numpyro import JeansLikelihoodModel


@contextmanager
def precision(enabled):
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def params(**overrides):
    return dict(dict(rs_pc=1000., rhos_Msunpc3=.01, alpha=1., beta=4., gamma=2.5,
                     r_t_pc=10000.), **overrides)


@lru_cache(maxsize=None)
def reference_mass(x, alpha, beta, gamma):
    # QUADPACK's algebraically weighted adaptive rule treats the cusp directly;
    # it does not use the implementation's power substitution or fixed nodes.
    p, q = 3 - gamma, (beta - gamma) / alpha
    c = min(x, 1.)
    inner = c**p * quad(lambda u: (1 + (c*u)**alpha)**(-q), 0, 1,
                        weight="alg", wvar=(p-1, 0), epsabs=1e-10,
                        epsrel=1e-10, limit=300)[0]
    outer = quad(lambda t: np.exp(p*t-q*np.logaddexp(0, alpha*t)), 0,
                 np.log(max(x, 1.)), epsabs=1e-10, epsrel=1e-10)[0]
    return 4*np.pi*(inner+outer)


@pytest.mark.parametrize("x64,rtol", [(True, 1e-6), (False, 5e-5)])
def test_mass_accuracy_grid(x64, rtol):
    with precision(x64):
        radii = np.array([1e-6, .01, 1., 100., 1e6])
        model = functional.ZhaoModel()
        evaluate = jax.jit(lambda r, p: model.enclosed_mass(r, params=p))
        for alpha, beta, gamma in product([.5, 1., 3., 5.], [2., 3., 4., 8.], [0., 1., 2.5, 2.99]):
            p = params(rs_pc=1., rhos_Msunpc3=1., r_t_pc=1e7, alpha=alpha, beta=beta, gamma=gamma)
            expected = [reference_mass(x, alpha, beta, gamma) for x in radii]
            np.testing.assert_allclose(evaluate(jnp.asarray(radii), p), expected, rtol=rtol)
            if x64:
                np.testing.assert_allclose(classical.ZhaoModel(**p).enclosed_mass(radii),
                                           expected, rtol=rtol)


@pytest.mark.parametrize("gamma", [0., 1., 2.5, 2.9, 2.99])
def test_dehnen_closed_form_and_truncation(gamma):
    with precision(True):
        p = params(gamma=gamma)
        radii = np.array([0., 1e-3, 10., 1000., 10000., 20000.])
        x = np.minimum(radii, p['r_t_pc']) / p['rs_pc']
        expected = 4*np.pi*.01*1000**3/(3-gamma)*(x/(1+x))**(3-gamma)
        for model in (classical.ZhaoModel(**p), functional.ZhaoModel()):
            kw = {} if isinstance(model, classical.ZhaoModel) else {'params': p}
            np.testing.assert_allclose(model.enclosed_mass(radii, **kw), expected, rtol=1e-8)
            assert np.shape(model.enclosed_mass(1000., **kw)) == ()


@pytest.mark.parametrize("beta,gamma", [(2., 0.), (3., 0.), (3., 1.), (4., 2.5)])
def test_explicit_analytic_domain_and_nfw_limit(beta, gamma):
    with precision(True):
        p = params(beta=beta, gamma=gamma)
        r = jnp.array([0., .001, 100., 1000., 1e5])
        model = functional.ZhaoModel()
        result = jax.jit(lambda r: model.enclosed_mass(r, params=p, method='analytic'))(r)
        np.testing.assert_allclose(result, classical.ZhaoModel(**p).enclosed_mass(r), rtol=1e-7)
        if beta == 3 and gamma == 0:
            assert float(result[2]) == pytest.approx(33785.62865245551, rel=1e-8)


@pytest.mark.parametrize("x64,rtol", [(True, 3e-5), (False, .015)])
def test_numeric_parameter_gradients_match_finite_differences(x64, rtol):
    with precision(x64):
        p = params(alpha=.8, beta=3., gamma=2.9)
        names = tuple(p)
        values = jnp.array([p[k] for k in names])
        def objective(v):
            return jnp.sum(jnp.log(functional.ZhaoModel().enclosed_mass(
                jnp.array([200., 3000., 30000.]), params=dict(zip(names, v)))))
        actual = np.asarray(jax.jit(jax.grad(objective))(values))
        # Independent float64 classical evaluation, with alpha tighter Gauss rule.
        def reference(v):
            return np.log(classical.ZhaoModel(**dict(zip(names, v))).enclosed_mass(
                np.array([200., 3000., 30000.]), n_steps=256)).sum()
        v = np.asarray(values, dtype=float)
        expected = []
        for i in range(len(names)):
            delta = max(abs(v[i]), .01)*1e-5
            offset = np.eye(len(names))[i]*delta
            expected.append((reference(v+offset)-reference(v-offset))/(2*delta))
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1e-7)


def dsph(module, p):
    if module is classical:
        return module.DSphModel(submodels=dict(
            StellarModel=module.PlummerModel(re_pc=220.), DMModel=module.ZhaoModel(**p),
            AnisotropyModel=module.ConstantAnisotropyModel(beta_ani=0.)))
    return module.DSphModel(submodels=dict(
        StellarModel=module.PlummerModel(), DMModel=module.ZhaoModel(),
        AnisotropyModel=module.ConstantAnisotropyModel()))


@pytest.mark.parametrize("bad", [dict(alpha=0.), dict(gamma=3.), dict(gamma=3.1), dict(beta=np.nan),
                                  dict(rs_pc=0.), dict(rhos_Msunpc3=-1.), dict(r_t_pc=-1.)])
def test_invalid_domains_are_not_zero_mass(bad):
    p = params(**bad)
    with pytest.raises(ValueError, match='Zhao mass domain'):
        classical.ZhaoModel(**p).enclosed_mass(100.)
    for method in ('auto', 'analytic'):
        evaluate = lambda r: functional.ZhaoModel().enclosed_mass(r, params=p, method=method)
        assert np.isnan(evaluate(100.))
        assert np.isnan(jax.jit(evaluate)(100.))


@pytest.mark.parametrize("backend", ['kernel', 'abel'])
def test_los_mass_resolution_and_invalid_likelihood(backend):
    with precision(True):
        p = dict(params(gamma=2.5), re_pc=220., beta_ani=0., vmem_kms=0.)
        model = dsph(functional, p)
        r = jnp.array([50., 100., 300.])
        kwargs = dict(solver=backend, n_u=257, n_r=1024, u_max=2000.)
        low = model.sigmalos2(r, params=p, dm_mass_n_steps=64, **kwargs)
        high = model.sigmalos2(r, params=p, dm_mass_n_steps=128, **kwargs)
        np.testing.assert_allclose(low, high, rtol=1e-5)
        refined = model.sigmalos2(r, params=p, dm_mass_n_steps=128,
                                  **dict(kwargs, n_u=513, n_r=2048))
        np.testing.assert_allclose(high, refined, rtol=.004)
        eager = model.sigmalos2(r, params=p, dm_mass_n_steps=128, jit=False, **kwargs)
        np.testing.assert_allclose(high, eager, rtol=1e-10)
        assert np.all(np.asarray(high) > 0)
        invalid = dict(p, gamma=3.)
        assert np.isnan(model.sigmalos2(r, params=invalid, **kwargs)).all()
        likelihood = JeansLikelihoodModel(model, [], parameter_postprocess=lambda _: invalid,
                                          sigmalos2_kwargs=kwargs)
        value, _ = log_density(likelihood, (r, jnp.zeros(3), jnp.ones(3)), {}, {})
        assert np.isneginf(value)


def test_classical_los_beta_domain_and_bad_mass():
    p = params(beta=3., gamma=0.)
    model = dsph(classical, p)
    r = np.array([50., 100., 300.])
    expected = model.sigmalos2(r)
    assert np.all(expected > 0)
    assert np.all(model.sigmar2(r) > 0)
    with precision(True):
        actual = dsph(functional, p).sigmalos2(
            jnp.asarray(r), params=dict(p, re_pc=220., beta_ani=0.),
            n_u=513, u_max=2000.)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)
    model['DMModel'].enclosed_mass = lambda r: np.full_like(r, np.nan)
    with pytest.raises(ValueError, match='enclosed mass'):
        model.sigmalos2(r)


def test_resolution_and_radius_validation():
    p = params()
    for value in [0, 7, 12.5]:
        with pytest.raises((TypeError, ValueError)):
            functional.ZhaoModel().enclosed_mass(100., params=p, n_steps=value)
    for radius in [-1., np.nan]:
        with pytest.raises(ValueError):
            classical.ZhaoModel(**p).enclosed_mass(radius)
        assert np.isnan(functional.ZhaoModel().enclosed_mass(radius, params=p))


@pytest.mark.parametrize("x64", [True, False])
def test_explicit_analytic_fallback_at_saturated_beta_argument(x64):
    with precision(x64):
        p = params(rs_pc=1., rhos_Msunpc3=1., r_t_pc=1e7, alpha=5., beta=3.01, gamma=1.)
        result = functional.ZhaoModel().enclosed_mass(jnp.array(1e6), params=p, method='analytic')
        assert float(result) == pytest.approx(reference_mass(1e6, 5., 3.01, 1.), rel=5e-5)
