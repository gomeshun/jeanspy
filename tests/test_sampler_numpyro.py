import pathlib
import sys
import importlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import arviz as az
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
import xarray as xr
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS

from jeanspy.model_numpyro import ConstantAnisotropyModel, DSphModel, NFWModel, PlummerModel
from jeanspy.sampler_numpyro import JeansLikelihoodModel, NumPyroSampler, ParameterSpec


def _simple_normal_model(y=None):
    x = numpyro.sample("x", dist.Normal(0.0, 1.0))
    numpyro.sample("obs", dist.Normal(x, 1.0), obs=y)


def _require_storage_backend(backend: str):
    package_name = {
        "zarr": "zarr",
        "h5netcdf": "h5netcdf",
        "netcdf4": "netCDF4",
    }[backend]
    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError:
        pytest.skip(f"Optional ArviZ storage backend dependency missing: {package_name}")


def test_jeans_likelihood_model_records_transformed_parameters():
    dsph = DSphModel(
        submodels={
            "StellarModel": PlummerModel(),
            "DMModel": NFWModel(),
            "AnisotropyModel": ConstantAnisotropyModel(),
        }
    )

    true_params = {
        "re_pc": 200.0,
        "rs_pc": 1200.0,
        "rhos_Msunpc3": 1e-2,
        "r_t_pc": 8000.0,
        "beta_ani": 0.2,
        "vmem_kms": 0.0,
    }
    R_pc = jnp.geomspace(5.0, 200.0, 4)
    e_vlos_kms = jnp.full_like(R_pc, 2.0)
    sigma2 = dsph.sigmalos2(R_pc, params=true_params, n_u=32, u_max=400.0)
    vlos_kms = true_params["vmem_kms"] + jnp.sqrt(sigma2 + e_vlos_kms**2) * jnp.array([0.1, -0.2, 0.3, -0.1])

    model = JeansLikelihoodModel(
        dsph,
        [
            ParameterSpec.exp("log_re", dist.Normal(jnp.log(200.0), 0.2), param_name="re_pc"),
            ParameterSpec.exp("log_rs", dist.Normal(jnp.log(1200.0), 0.2), param_name="rs_pc"),
            ParameterSpec.exp(
                "log_rhos",
                dist.Normal(jnp.log(1e-2), 0.2),
                param_name="rhos_Msunpc3",
            ),
            ParameterSpec.exp("log_r_t", dist.Normal(jnp.log(8000.0), 0.2), param_name="r_t_pc"),
            ParameterSpec("beta_ani", dist.Uniform(-0.5, 0.8)),
            ParameterSpec("vmem_kms", dist.Normal(0.0, 30.0)),
        ],
        sigmalos2_kwargs={"n_u": 32, "u_max": 400.0},
    )

    model_trace = trace(seed(model, jax.random.PRNGKey(0))).get_trace(
        R_pc=R_pc,
        vlos_kms=vlos_kms,
        e_vlos_kms=e_vlos_kms,
    )

    assert model_trace["log_re"]["type"] == "sample"
    assert model_trace["re_pc"]["type"] == "deterministic"
    assert model_trace["rs_pc"]["type"] == "deterministic"
    assert model_trace["vlos"]["is_observed"] is True


@pytest.mark.parametrize(
    ("storage_backend", "expected_suffix"),
    [("zarr", ".zarr"), ("h5netcdf", ".nc"), ("netcdf4", ".nc")],
)
def test_numpyro_sampler_checkpoint_resume_and_chunk_loading(tmp_path, storage_backend, expected_suffix):
    _require_storage_backend(storage_backend)
    observed = jnp.array([0.25, -0.1, 0.05], dtype=jnp.float32)

    first = NumPyroSampler(
        MCMC(NUTS(_simple_normal_model), num_warmup=5, num_samples=4, num_chains=1, progress_bar=False),
        output_dir=tmp_path,
        storage_backend=storage_backend,
        async_writes=True,
    )
    first_result = first.run(jax.random.PRNGKey(0), y=observed)
    second_result = first.run(jax.random.PRNGKey(1), y=observed)

    combined = first.load_samples(combine=True)
    assert isinstance(combined, xr.DataTree)
    assert combined.children["posterior"].dataset.sizes["draw"] == 8
    assert first.checkpoint_path.exists()
    assert len(first.list_chunk_paths()) == 2
    assert all(path.suffix == expected_suffix for path in first.list_chunk_paths())
    assert first_result.resumed is False
    assert second_result.resumed is True
    summary = az.summary(combined, var_names=["x"])
    assert not summary.empty

    resumed = NumPyroSampler(
        MCMC(NUTS(_simple_normal_model), num_warmup=5, num_samples=4, num_chains=1, progress_bar=False),
        output_dir=tmp_path,
        storage_backend=storage_backend,
        async_writes=True,
    )
    resumed.load_checkpoint()
    resumed_result = resumed.run(jax.random.PRNGKey(2), y=observed, resume=True)
    combined_after_resume = resumed.load_samples(combine=True)
    assert isinstance(combined_after_resume, xr.DataTree)

    assert combined_after_resume.children["posterior"].dataset.sizes["draw"] == 12
    assert resumed_result.resumed is True
    first.close()
    resumed.close()