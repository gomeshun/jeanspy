"""Exercise the public classical inference composition with explicit priors."""
import numpy as np
import pandas as pd
import pytest

from jeanspy.model import (
    ConstantAnisotropyModel, DSphModel, FlatPriorModel, NFWModel,
    PhotometryPriorModel, PlummerModel, SimpleDSphEstimationModel,
    get_default_estimation_model,
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


@pytest.mark.parametrize("failure", ["raises", "shape", "nan", "dependent", "posterior"])
def test_failed_reset_preserves_stored_chain(tmp_path, failure):
    model = _make_model(tmp_path, "dataframe")
    sampler = Sampler(model, _initial_state, nwalkers=16, prefix=f"{tmp_path}/")
    sampler.run_mcmc(4, 1, enable_convergence_check=False)
    chain, log_prob, blobs = (sampler.get_chain(), sampler.get_log_prob(), sampler.get_blobs())
    calls = []

    def invalid(n):
        calls.append(n)
        if n is None:
            return _initial_state(None)
        if failure == "raises":
            raise ValueError("generator failed")
        coords = _initial_state(n)
        if failure == "shape":
            return coords[:-1]
        if failure == "nan":
            coords[0, 0] = np.nan
        if failure == "dependent":
            coords[:] = coords[0]
        if failure == "posterior":
            coords[:, 0] += 1000.
        return coords

    with pytest.raises((ValueError, RuntimeError)):
        sampler.run_mcmc(2, 1, reset=True, p0_generator=invalid)
    assert calls == [None, 16]
    np.testing.assert_array_equal(sampler.get_chain(), chain)
    np.testing.assert_array_equal(sampler.get_log_prob(), log_prob)
    np.testing.assert_array_equal(sampler.get_blobs(), blobs)
    sampler.run_mcmc(1, 1, enable_convergence_check=False)
    assert sampler.get_chain().shape == (5, 16, 6)


def test_successful_reset_generates_replacement_once(tmp_path):
    model = _make_model(tmp_path, "dataframe")
    sampler = Sampler(model, _initial_state, nwalkers=16, prefix=f"{tmp_path}/")
    sampler.run_mcmc(4, 1, enable_convergence_check=False)
    calls = []

    def generator(n):
        calls.append(n)
        return _initial_state(n)

    sampler.run_mcmc(4, 1, reset=True, p0_generator=generator, enable_convergence_check=False)
    assert calls == [None, 16]
    assert sampler.get_chain().shape == (4, 16, 6)


def _data(n=12):
    return pd.DataFrame(dict(R_pc=np.linspace(10., 300., n),
                             vlos_kms=np.linspace(-10., 10., n),
                             e_vlos_kms=np.full(n, 2.)))


@pytest.mark.parametrize("config_kind", ["dataframe", "path"])
def test_default_model_samples_finite_posterior_and_restarts(tmp_path, classical_prior_config, config_kind):
    config = classical_prior_config
    if config_kind == "path":
        config = tmp_path / "prior.csv"
        classical_prior_config.to_csv(config)
    model = get_default_estimation_model(_data(), 2.3, .1, config=config)
    old_random = np.random.get_state()
    np.random.seed(53)
    try:
        samples = model.sample(64)
        assert samples.shape == (64, 6)
        assert np.isfinite([model.lnposterior(p) for p in samples]).all()
        sampler = Sampler(model, model.sample, nwalkers=16, prefix=f"{tmp_path}/")
        sampler.run_mcmc(4, 1, enable_convergence_check=False)
        chain = sampler.get_chain().copy()
        reloaded = get_default_estimation_model(_data(), 2.3, .1, config=config)
        resumed = Sampler(reloaded, reloaded.sample, nwalkers=16, prefix=f"{tmp_path}/")
        resumed.run_mcmc(3, 1, enable_convergence_check=False)
        assert resumed.get_chain().shape == (7, 16, 6)
        np.testing.assert_array_equal(resumed.get_chain()[:4], chain)
        assert np.isfinite(resumed.get_log_prob()).all()
        assert np.isfinite(resumed.get_blobs()["lnl"]).all()
    finally:
        np.random.set_state(old_random)


def test_missing_prior_creates_correct_template_and_fails_early(tmp_path, classical_prior_config):
    path = tmp_path / "prior.csv"
    with pytest.raises(ValueError, match="Created prior template"):
        get_default_estimation_model(_data(), 2.3, .1, config=path)
    template = pd.read_csv(path, index_col=0)
    assert template.index.tolist() == classical_prior_config.index.tolist()
    assert template.isna().all().all()
    # Existing incomplete config is never overwritten on retry.
    content = path.read_bytes()
    with pytest.raises(ValueError, match="explicit finite prior bounds"):
        get_default_estimation_model(_data(), 2.3, .1, config=path)
    assert path.read_bytes() == content


@pytest.mark.parametrize("bad", ["missing", "extra", "order", "physical", "substring", "duplicate"])
def test_prior_schema_rejected_before_inference(classical_prior_config, bad):
    config = classical_prior_config
    if bad == "missing":
        config = config.iloc[:-1]
    elif bad == "extra":
        config.loc["extra"] = [0., 1.]
    elif bad == "order":
        config = config.iloc[::-1]
    elif bad == "physical":
        config = config.rename(index={"log10_re_pc": "re_pc"})
    elif bad == "substring":
        config = config.rename(index={"log10_re_pc": "arbitrary_log10_re_pc"})
    else:
        config.index = [config.index[0]] * len(config)
    with pytest.raises(ValueError, match="names"):
        get_default_estimation_model(_data(), 2.3, .1, config=config)


@pytest.mark.parametrize("lower,upper", [(np.nan, 1.), (-np.inf, 1.), (0., np.inf), (1., 1.), (2., 1.)])
def test_non_sampleable_bounds_fail_early(classical_prior_config, lower, upper):
    classical_prior_config.loc["vmem_kms"] = [lower, upper]
    with pytest.raises(ValueError, match="finite prior bounds"):
        get_default_estimation_model(_data(), 2.3, .1, config=classical_prior_config)


def test_data_reset_preserves_explicit_priors_and_invalidates_wbic(classical_prior_config):
    model = get_default_estimation_model(_data(12), 2.3, .1, config=classical_prior_config)
    assert model.inverse_temparature == pytest.approx(1 / np.log(12))
    model.reset_data(_data(24).assign(vlos_kms=100.))
    assert model.inverse_temparature == pytest.approx(1 / np.log(24))
    pd.testing.assert_frame_equal(model["FlatPriorModel"].data, classical_prior_config)


def test_empirical_velocity_bounds_stay_synchronized(classical_prior_config):
    model = get_default_estimation_model(_data(), 2.3, .1, config=classical_prior_config,
                                         vmem_prior_from_data=True)
    model.reset_data(_data().assign(vlos_kms=np.linspace(90., 110., 12)))
    prior = model["FlatPriorModel"]
    assert prior.lower[0] == 90.
    assert prior.upper[0] == 110.
    sampled = model.sample(100)
    assert ((sampled[:, 0] >= 90.) & (sampled[:, 0] <= 110.)).all()
    assert np.isfinite([prior._lnprior(p) for p in sampled]).all()
    assert classical_prior_config.loc["vmem_kms", "lower"] == -30.
    before = {k: v.copy() for k, v in model.data.items()}
    with pytest.raises(ValueError, match="finite prior bounds"):
        model.reset_data(_data().assign(vlos_kms=100.))
    for name, values in before.items():
        np.testing.assert_array_equal(model.data[name], values)


def test_truncated_photometry_samples_respect_flat_support(classical_prior_config):
    classical_prior_config.loc["log10_re_pc"] = [2.39, 2.40]
    model = get_default_estimation_model(_data(), 2.3, .1, config=classical_prior_config)
    for size in (None, 200, (2, 3)):
        sampled = model.sample(size)
        assert ((sampled[..., 1] >= 2.39) & (sampled[..., 1] <= 2.40)).all()
    for invalid in ([0.] * 5, [0.] * 7, np.zeros((1, 6))):
        with pytest.raises(ValueError, match="shape"):
            model.lnposterior(invalid)
