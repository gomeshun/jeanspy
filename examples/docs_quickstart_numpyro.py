"""Run the same spherical inference example with JAX and NumPyro NUTS."""
# imports-start
import os
os.environ.setdefault("JEANSPY_JAX_PLATFORM", "cpu")
os.environ.setdefault("JEANSPY_JAX_ENABLE_X64", "true")

# Import JeansPy before JAX so its precision/device settings take effect.
from jeanspy.model_numpyro import (
    DSphModel, PlummerModel, ZhaoModel, ConstantAnisotropyModel,
)
from jeanspy.sampler_numpyro import JeansLikelihoodModel, NumPyroSampler, ParameterSpec
import jax
import numpy as np
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from pathlib import Path
import argparse
import arviz as az
from docs_quickstart_data import data
from docs_quickstart_plots import plot_data, plot_posterior
R_pc, vlos_kms, e_vlos_kms = (data[k] for k in ("R_pc", "vlos_kms", "e_vlos_kms"))

# imports-end

# model-start
model = DSphModel(submodels={
    "StellarModel": PlummerModel(),
    "DMModel": ZhaoModel(),
    "AnisotropyModel": ConstantAnisotropyModel(),
})
fixed = dict(re_pc=200., a=2., b=4., g=.5, r_t_pc=np.inf)
truth = dict(**fixed, rs_pc=500., rhos_Msunpc3=.1, beta_ani=0.)
options = dict(backend="kernel", n_u=128, n_kernel=32)
sigma_kms = np.sqrt(model.sigmalos2(R_pc, params=truth, **options))
# model-end

# inference-start
def physical_parameters(sampled):
    return {**fixed, **sampled}


likelihood = JeansLikelihoodModel(
    model,
    [ParameterSpec.pow10("log10_rs_pc", dist.Uniform(2.4, 3.2), param_name="rs_pc"),
     ParameterSpec.pow10("log10_rhos_Msunpc3", dist.Uniform(-2., -.3),
                         param_name="rhos_Msunpc3"),
     ParameterSpec("beta_ani", dist.Uniform(-.5, .5)),
     ParameterSpec("vmem_kms", dist.Uniform(-20., 20.))],
    parameter_postprocess=physical_parameters,
    sigmalos2_kwargs=options,
)


def build_mcmc(num_warmup, num_samples):
    kernel = NUTS(likelihood, target_accept_prob=.85,
                  init_strategy=init_to_value(values={
                      "log10_rs_pc": np.log10(500.), "log10_rhos_Msunpc3": -1.,
                      "beta_ani": 0., "vmem_kms": 0.}))
    return MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=2, chain_method="sequential", progress_bar=False)


def run(output_dir, num_warmup=200, num_samples=256):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savetxt(output_dir / "observations.csv", np.column_stack(list(data.values())),
               delimiter=",", header=",".join(data), comments="")
    plot_data(data, output_dir)
    # checkpoint-start
    with NumPyroSampler(build_mcmc(num_warmup, num_samples), output_dir=output_dir,
                         storage_backend="h5netcdf", async_writes=False) as first_sampler:
        first = first_sampler.run(jax.random.PRNGKey(42), **data, resume=False)
        original = first_sampler.load_samples()["posterior"].ds.load()
        print("First checkpoint saved:", first.checkpoint_path.is_file())
    # A fresh sampler object restores the state and skips the completed warmup.
    with NumPyroSampler(build_mcmc(num_warmup, num_samples), output_dir=output_dir,
                         storage_backend="h5netcdf", async_writes=False) as resumed_sampler:
        resumed = resumed_sampler.run(jax.random.PRNGKey(43), **data, resume=True)
        idata = resumed_sampler.load_samples(combine=True)
        posterior = idata["posterior"].ds
        print("Resumed from checkpoint:", resumed.resumed)
        print("Stored chunks:", len(resumed_sampler.list_chunk_paths()))
        print("Stored samples per parameter:", posterior["beta_ani"].shape)
        for name in original.data_vars:
            np.testing.assert_array_equal(posterior[name].values[:, :num_samples],
                                          original[name].values)
        # checkpoint-end
        # diagnostics-start
        names = ["log10_rs_pc", "log10_rhos_Msunpc3", "beta_ani", "vmem_kms"]
        print(az.summary(idata, var_names=names).round(3).to_string())
        print("Divergences:", int(idata["sample_stats"]["diverging"].sum()))
        chain = np.stack([posterior[n].values for n in names], axis=-1)
        plot_posterior(chain, names, [np.log10(500.), -1., 0., 0.], output_dir)
        # diagnostics-end
        assert np.all(np.isfinite(chain))
        return idata
# inference-end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args().output_dir)
