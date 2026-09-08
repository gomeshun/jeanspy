"""Exercise the public classical inference composition with explicit priors."""
import numpy as np
import pandas as pd
import pytest

from jeanspy.model import (
    ConstantAnisotropyModel, DSphModel, FlatPriorModel, NFWModel,
    PhotometryPriorModel, PlummerModel, SimpleDSphEstimationModel,
)
from jeanspy.sampler import Sampler


def _make_model(tmp_path, config_kind):
    config = pd.DataFrame(
        {"lower": [-30., 2., 2.5, -3., 3.5, -.3],
         "upper": [30., 2.6, 3.5, -1., 4.5, .3]},
        index=["vmem_kms", "log10_re_pc", "log10_rs_pc",
               "log10_rhos_Msunpc3", "log10_r_t_pc", "bfunc_beta_ani"],
    )
    if config_kind == "path":
        path = tmp_path / "prior.csv"
        config.to_csv(path)
        config = path
    data = pd.DataFrame({"R_pc": np.linspace(10., 300., 12),
                         "vlos_kms": np.linspace(-10., 10., 12),
                         "e_vlos_kms": np.full(12, 2.)})
    return SimpleDSphEstimationModel(args_load_data=[data], submodels={
        "DSphModel": DSphModel(submodels={
            "StellarModel": PlummerModel(), "DMModel": NFWModel(),
            "AnisotropyModel": ConstantAnisotropyModel(),
        }),
        "FlatPriorModel": FlatPriorModel(config),
        "PhotometryPriorModel": PhotometryPriorModel(2.3, .1),
    })


def _initial_state(nwalkers):
    center = np.array([0., 2.3, 3., -2., 4., 0.])
    if nwalkers is None:
        return center
    return center + np.random.default_rng(42).normal(0., .01, (nwalkers, 6))


@pytest.mark.parametrize("config_kind", ["dataframe", "path"])
def test_classical_inference_runs_resumes_and_resets(tmp_path, config_kind):
    model = _make_model(tmp_path, config_kind)
    assert not hasattr(model, "dsph_name")
    assert np.isfinite(model.lnposterior(_initial_state(None))).all()
    if config_kind == "dataframe":
        assert model["FlatPriorModel"].fname_config is None

    sampler = Sampler(model, _initial_state, nwalkers=16, prefix=f"{tmp_path}/")
    sampler.run_mcmc(4, 1, enable_convergence_check=False)
    assert sampler.get_chain().shape == (4, 16, 6)
    assert np.isfinite(sampler.get_log_prob()).all()
    # Recreate the wrapper to check persistence and resumption.
    resumed = Sampler(model, _initial_state, nwalkers=16, prefix=f"{tmp_path}/")
    resumed.run_mcmc(3, 1, enable_convergence_check=False)
    assert resumed.get_chain().shape == (7, 16, 6)
    resumed.run_mcmc(2, 1, reset=True, enable_convergence_check=False)
    assert resumed.get_chain().shape == (2, 16, 6)
    assert np.isfinite(resumed.get_blobs()["lnl"]).all()
