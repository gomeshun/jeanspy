"""Independent reproductions of the 2026-09-10 release audit failures."""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace
from numpyro.infer.util import log_density
from scipy.integrate import quad

from jeanspy import model as classical
from jeanspy import model_numpyro as functional
from jeanspy.sampler_numpyro import JeansLikelihoodModel, ParameterSpec


PARAMS = dict(rs_pc=1000., rhos_Msunpc3=.01, r_t_pc=10000.)


@contextmanager
def precision(enabled):
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', enabled)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


@pytest.mark.parametrize("x64", [False, True])
def test_integer_numeric_nfw_mass_and_zero_gradient(x64):
    with precision(x64):
        dm = functional.NFWModel()
        radii = jnp.array([0, 100, 1000, 20000])
        numeric = jax.jit(lambda r: dm.enclosed_mass(r, params=PARAMS, method="numeric", n_steps=1024))(radii)
        expected = classical.NFWModel(**PARAMS).enclosed_mass(np.asarray(radii))
        np.testing.assert_allclose(numeric, expected, rtol=5e-5)
        gradient = jax.grad(lambda r: dm.enclosed_mass(r, params=PARAMS, method="numeric"))(0.)
        assert np.isfinite(gradient)


@pytest.mark.parametrize("bad", [dict(rhos_Msunpc3=np.nan), dict(rhos_Msunpc3=np.inf),
                                  dict(rhos_Msunpc3=-1.), dict(rs_pc=0.), dict(r_t_pc=-1.)])
@pytest.mark.parametrize("method", ["numeric", "analytic"])
def test_invalid_nfw_is_rejected_by_jitted_likelihood(bad, method):
    params = dict(PARAMS, re_pc=200., beta_ani=0., vmem_kms=0.)
    params.update(bad)
    dm = functional.NFWModel()
    mass = jax.jit(lambda r: dm.enclosed_mass(r, params=params, method=method))
    assert np.isnan(mass(jnp.array([0., 100.]))).all()
    dsph = functional.DSphModel(submodels={
        "StellarModel": functional.PlummerModel(), "DMModel": dm,
        "AnisotropyModel": functional.ConstantAnisotropyModel(),
    })
    model = JeansLikelihoodModel(dsph, [], parameter_postprocess=lambda _: params,
                                  sigmalos2_kwargs={"dm_mass_method": method, "n_u": 32})
    evaluate = jax.jit(lambda: log_density(model, (jnp.array([100.]), jnp.zeros(1), jnp.ones(1)), {}, {})[0])
    assert np.isneginf(evaluate())


class ConstantVariance:
    def sigmalos2(self, R, **kwargs):
        return jnp.ones_like(R) * 4.


def likelihood(**kwargs):
    return JeansLikelihoodModel(ConstantVariance(), [], velocity_mean=lambda _: 0., **kwargs)


@pytest.mark.parametrize("shapes", [((3,), (3, 1), (3,)), ((3,), (2,), (3,)),
                                     ((), (), ()), ((0,), (0,), (0,))])
def test_likelihood_rejects_broadcasting_observations(shapes):
    arrays = tuple(jnp.ones(s) for s in shapes)
    with pytest.raises(ValueError, match="matching nonempty 1-D"):
        log_density(likelihood(), arrays, {}, {})


@pytest.mark.parametrize("which,value", [(0, 0.), (0, -1.), (0, np.inf),
                                         (1, np.nan), (2, -1.), (2, np.inf)])
def test_dynamic_invalid_observations_have_negative_infinite_log_density(which, value):
    arrays = [jnp.array([10., 20.]), jnp.zeros(2), jnp.ones(2)]
    arrays[which] = arrays[which].at[0].set(value)
    model = likelihood()
    evaluate = jax.jit(lambda r, v, e: log_density(model, (r, v, e), {}, {})[0])
    assert np.isneginf(evaluate(*arrays))


def test_valid_likelihood_has_exactly_one_term_per_star():
    R, v, e = jnp.array([10., 20., 30.]), jnp.array([-1., 0., 2.]), jnp.ones(3)
    actual, model_trace = log_density(likelihood(), (R, v, e), {}, {})
    expected = dist.Normal(0., jnp.sqrt(5.)).log_prob(v).sum()
    assert model_trace['vlos']['fn'].log_prob(v).shape == (3,)
    assert actual == pytest.approx(float(expected))


@pytest.mark.parametrize("bounds", [(0., 1.), (2., 1.), (1., np.inf), (np.nan, 1.)])
def test_invalid_variance_bounds(bounds):
    with pytest.raises(ValueError, match="sigma2_bounds"):
        likelihood(sigma2_bounds=bounds)


@pytest.mark.parametrize("factory,expected", [(ParameterSpec.exp, np.e), (ParameterSpec.pow10, 10.)])
def test_transform_without_physical_name(factory, expected):
    spec = factory("raw", dist.Delta(1.))
    result = []
    def model():
        result.append(spec.sample())
    sites = trace(seed(model, 0)).get_trace()
    assert sites['raw']['type'] == 'sample'
    assert float(sites['raw_transformed']['value']) == pytest.approx(expected)
    assert result[0][0] == 'raw'
    assert float(result[0][1]) == pytest.approx(expected)


@pytest.mark.parametrize("g", [1.5, 1.6, 2.5, 3.])
@pytest.mark.parametrize("method", ["jfactor_ullio2016", "jfactor_ullio2016_simple"])
def test_divergent_zhao_annihilation_cusp_is_rejected(g, method):
    dm = classical.ZhaoModel(**PARAMS, a=1., b=4., g=g)
    with pytest.raises(ValueError, match="g < 1.5"):
        getattr(dm, method)(100000., .5)


def test_convergent_zhao_cusp_reference():
    # An independently weighted quadrature resolves the singular origin.
    g = 1.4
    dm = classical.ZhaoModel(**PARAMS, a=1., b=4., g=g)
    upper = 100000. * np.sin(np.deg2rad(.5)) / PARAMS['rs_pc']
    integral = quad(lambda x: (1+x)**(-2*(4-g)), 0., upper,
                    weight='alg', wvar=(2-2*g, 0.), epsabs=1e-10, epsrel=1e-10)[0]
    expected = classical.C_J * 4*np.pi*PARAMS['rhos_Msunpc3']**2 * PARAMS['rs_pc']**3 / 100000.**2 * integral
    assert dm.jfactor_ullio2016_simple(100000., .5) == pytest.approx(expected, rel=2e-8)


def test_quadrature_nonconvergence_is_an_error():
    class DivergentCustom(classical.DMModel):
        required_param_names = ['r_t_pc']
        required_models = {}
        def mass_density_3d(self, r):
            return np.asarray(r)**-2
    with pytest.raises(ValueError, match="J-factor quadrature failed"):
        DivergentCustom(r_t_pc=1000.).jfactor_ullio2016_simple(100000.)


def test_evans_transition_matches_high_precision_formula():
    dm = classical.NFWModel(**PARAMS)
    radii = np.r_[np.geomspace(.01, .85, 12), np.sqrt([.7999999, .8, .8000001]),
                 1 + np.r_[-np.geomspace(1e-12, .1, 15), 0., np.geomspace(1e-12, .1, 15)],
                 np.sqrt([1.1999999, 1.2, 1.2000001]), 1.3, 1.6]
    actual = dm.jfactor_evans2016(100000., np.rad2deg(radii / 100.))
    expected = []
    with mp.workdps(80):
        for radius in radii:
            y = mp.mpf(float(radius))
            delta = 1-y*y
            if y == 1:
                coeff = mp.pi-mp.mpf(38)/15
            else:
                x = mp.acosh(1/y)/mp.sqrt(delta) if y < 1 else mp.acos(1/y)/mp.sqrt(-delta)
                coeff = (2*y*(7*y-4*y**3+3*mp.pi*delta**2)+6*(2*delta**3-2*delta-y**4)*x)/(6*delta**2)
            expected.append(float(coeff)*classical.C_J*2*np.pi*.01**2*1000.**3/100000.**2)
    np.testing.assert_allclose(actual, expected, rtol=2e-11)
    assert np.isfinite(dm.jfactor_evans2016(100000., np.rad2deg(.01)))


def test_truncated_surface_density_normalizes_over_all_radii():
    model = classical.PlummerModel(re_pc=200.)
    pdf = lambda r: 2*np.pi*r*model.density_2d_truncated(r, 200.)
    assert quad(pdf, 0, 200.)[0] + quad(pdf, 200., np.inf)[0] == pytest.approx(1.)
    assert model.density_2d_truncated(400., 200.) == 0.
    for cutoff in [0., -1., np.inf, np.nan]:
        with pytest.raises(ValueError):
            model.density_2d_truncated(10., cutoff)


def test_uniform_disk_density_and_cdf_support():
    model = classical.Uniform2dModel(Rmax_pc=200.)
    r = np.array([-np.inf, -10., 0., 100., 200., 400., np.inf])
    np.testing.assert_allclose(model.cdf_R(r), [0., 0., 0., .25, 1., 1., 1.])
    np.testing.assert_allclose(model.density_2d(r)[[0, 1, 5, 6]], 0.)
    assert quad(lambda x: 2*np.pi*x*model.density_2d(x), 0., 200.)[0] == pytest.approx(1.)

