"""NumPyro axisymmetric likelihood, derivatives and real checkpoint workflows."""
import numpy as np
import pytest
from scipy.stats import norm

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import log_density

from jeanspy.axisymmetric import AxisymmetricDSphModel as ClassicalModel
from jeanspy.axisymmetric_numpyro import AxisymmetricDSphModel
from jeanspy.sampler_numpyro import AxisymmetricJeansLikelihoodModel, NumPyroSampler, ParameterSpec


FIXED = dict(re_pc=300., rs_pc=500., q=.7, Q=.8, alpha=2., beta=3.,
             gamma=.5, beta_z=-.3, inclination=1.1)
DATA = dict(x_pc=np.array([0., 100., -200.]), y_pc=np.array([0., -50., 150.]),
            vlos_kms=np.array([1., -3., 5.]), e_vlos_kms=np.array([0., 1., 2.]))


@pytest.fixture(autouse=True)
def precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def make_model(**kwargs):
    options = dict(dsph_model=AxisymmetricDSphModel(16, 16, 16),
        parameter_specs=[ParameterSpec.pow10("log10_rhos", dist.Uniform(-1.5, -.5), param_name="rhos_Msunpc3"),
                         ParameterSpec("vmem_kms", dist.Normal(0., 20.))], fixed_params=FIXED)
    options.update(kwargs)
    return AxisymmetricJeansLikelihoodModel(**options)


def test_exact_unbinned_log_density_and_gradients():
    m = make_model()
    p = dict(log10_rhos=-1., vmem_kms=2.)
    f = lambda p: log_density(m, (), DATA, p)[0]
    sigma2 = ClassicalModel(16, 16, 16).sigmalos2(DATA["x_pc"], DATA["y_pc"],
                         params={**FIXED, "rhos_Msunpc3": .1})
    expected = norm.logpdf(DATA["vlos_kms"], 2., np.sqrt(sigma2+DATA["e_vlos_kms"]**2)).sum()
    expected += norm.logpdf(2., 0., 20.)  # uniform log-density is zero here
    np.testing.assert_allclose(f(p), expected, rtol=1e-11)
    gradient = jax.jit(jax.grad(f))(p)
    for name in p:
        h = 1e-4
        fd = (f({**p, name: p[name]+h})-f({**p, name: p[name]-h}))/(2*h)
        np.testing.assert_allclose(gradient[name], fd, rtol=1e-6, atol=1e-7)
    traced = trace(seed(m, jax.random.PRNGKey(0))).get_trace(**DATA)
    assert traced["rhos_Msunpc3"]["type"] == "deterministic"
    assert traced["vlos"]["is_observed"]


@pytest.mark.parametrize("name,value", [("x_pc", [0., np.nan, 2.]), ("y_pc", [0., 1., np.inf]),
    ("vlos_kms", [0., 1., np.nan]), ("e_vlos_kms", [0., -1., 0.])])
def test_invalid_observations_have_zero_density(name, value):
    data = {**DATA, name: np.array(value)}
    value, traced = log_density(make_model(), (), data, dict(log10_rhos=-1., vmem_kms=0.))
    assert float(value) == -np.inf
    assert np.isfinite(traced["vlos"]["fn"].scale).all()


@pytest.mark.parametrize("change", [dict(q=-1.), dict(beta_z=.99), dict(inclination=-1.)])
def test_invalid_physics_rejects_in_likelihood(change):
    m = make_model(fixed_params={**FIXED, **change})
    value = log_density(m, (), DATA, dict(log10_rhos=-1., vmem_kms=0.))[0]
    assert float(value) == -np.inf


def test_invalid_deprojection_and_explicit_orientation_prior():
    fixed = {k: v for k, v in FIXED.items() if k not in {"q", "inclination"}}
    fixed.update(q_projected=.8, rhos_Msunpc3=.1, vmem_kms=0.)
    m = make_model(fixed_params=fixed, parameter_specs=[
        ParameterSpec("cos_inclination", dist.Uniform(0., 1.), param_name="inclination", transform=jnp.arccos)])
    assert np.isfinite(log_density(m, (), DATA, dict(cos_inclination=.2))[0])
    assert float(log_density(m, (), DATA, dict(cos_inclination=.9))[0]) == -np.inf


def test_bad_shapes_and_duplicate_fixed_names_fail_early():
    with pytest.raises(ValueError, match="matching nonempty"):
        log_density(make_model(), (), {**DATA, "y_pc": [1.]}, dict(log10_rhos=-1., vmem_kms=0.))
    with pytest.raises(ValueError, match="disjoint"):
        make_model(fixed_params={**FIXED, "vmem_kms": 0.})
    with pytest.raises(ValueError, match="unique"):
        make_model(parameter_specs=[ParameterSpec("valid_observations", dist.Normal(0., 1.))])


def test_variance_bounds_reject_instead_of_clipping():
    m = make_model(sigma2_bounds=(1e-12, 1e-6))
    assert float(log_density(m, (), DATA, dict(log10_rhos=-1., vmem_kms=0.))[0]) == -np.inf


@pytest.mark.mcmc
@pytest.mark.parametrize("storage_backend", ["h5netcdf", "zarr"])
def test_real_axisymmetric_nuts_checkpoint_and_identity(tmp_path, storage_backend):
    def sampler(model):
        kernel = NUTS(model, max_tree_depth=3,
                      init_strategy=init_to_value(values=dict(log10_rhos=-1., vmem_kms=0.)))
        mcmc = MCMC(kernel, num_warmup=12, num_samples=6, num_chains=1, progress_bar=False)
        return NumPyroSampler(mcmc, output_dir=tmp_path, storage_backend=storage_backend, async_writes=False)

    with sampler(make_model()) as first:
        assert first.run(jax.random.PRNGKey(10), **DATA).resumed is False
        assert np.isfinite(first.mcmc.get_samples()["rhos_Msunpc3"]).all()

    with sampler(make_model()) as resumed:
        resumed.load_checkpoint()
        assert resumed.run(jax.random.PRNGKey(11), resume=True, **DATA).resumed is True
        assert resumed.load_samples()["posterior"].dataset.sizes["draw"] == 12
        paths = resumed.list_chunk_paths()
        metadata = resumed.metadata_path.read_bytes()
        checkpoint = resumed.checkpoint_path.read_bytes()
        with pytest.raises(ValueError, match="identity"):
            resumed.run(jax.random.PRNGKey(12), **{**DATA, "y_pc": DATA["y_pc"]+1.})
        assert paths == resumed.list_chunk_paths()
        assert metadata == resumed.metadata_path.read_bytes()
        assert checkpoint == resumed.checkpoint_path.read_bytes()

    with sampler(make_model(dsph_model=AxisymmetricDSphModel(24, 16, 16))) as changed:
        with pytest.raises(ValueError, match="identity"):
            changed.run(jax.random.PRNGKey(13), resume=True, **DATA)
