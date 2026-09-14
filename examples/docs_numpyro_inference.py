"""Two short NumPyro chunks verify persistence, not posterior convergence."""
from tempfile import TemporaryDirectory
import numpy as np
import jax
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from jeanspy.axisymmetric_numpyro import AxisymmetricDSphModel
from jeanspy.sampler_numpyro import (AxisymmetricJeansLikelihoodModel,
                                     NumPyroSampler, ParameterSpec)

# numpyro-inference-start
data = dict(x_pc=np.array([0., 100., -200.]), y_pc=np.array([0., -50., 150.]),
            vlos_kms=np.array([1., -3., 5.]), e_vlos_kms=np.array([1., 1., 2.]))
likelihood = AxisymmetricJeansLikelihoodModel(
    AxisymmetricDSphModel(16, 16, 16),
    [ParameterSpec.pow10("log10_rhos", dist.Uniform(-1.5, -.5),
                         param_name="rhos_Msunpc3"),
     ParameterSpec("vmem_kms", dist.Normal(0., 20.))],
    fixed_params=dict(re_pc=300., rs_pc=500., q=.7, Q=.8, alpha=2., beta=3.,
                      gamma=.5, beta_z=-.3, inclination=1.1),
)

def make_mcmc():
    return MCMC(NUTS(likelihood, max_tree_depth=3,
                      init_strategy=init_to_value(values={"log10_rhos": -1., "vmem_kms": 0.})),
                num_warmup=6, num_samples=4, num_chains=1, progress_bar=False)

with TemporaryDirectory() as directory:
    with NumPyroSampler(make_mcmc(), output_dir=directory, storage_backend="h5netcdf",
                         async_writes=False) as sampler:
        first = sampler.run(jax.random.PRNGKey(42), **data, resume=False)
        assert first.checkpoint_path.is_file() and first.chunk_path.is_file()
    # Reconstruct exactly the same model, data and sampler configuration.
    with NumPyroSampler(make_mcmc(), output_dir=directory, storage_backend="h5netcdf",
                         async_writes=False) as sampler:
        second = sampler.run(jax.random.PRNGKey(43), **data, resume=True)
        assert second.resumed
        posterior = sampler.load_samples()["posterior"].ds
        assert posterior.sizes["chain"] == 1 and posterior.sizes["draw"] == 8
# Production analyses need independent chains, longer warmup/draws and diagnostics.
# numpyro-inference-end
