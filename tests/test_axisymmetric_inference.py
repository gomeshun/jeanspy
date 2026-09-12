"""Observed-data, prior-coordinate and persisted classical inference contracts."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from jeanspy.axisymmetric import AxisymmetricDSphModel
from jeanspy.axisymmetric_inference import AxisymmetricDSphEstimationModel, AxisymmetricKinematicData
from jeanspy.model import PhotometryPriorModel
from jeanspy.sampler import Sampler
from jeanspy._sampling_identity import fingerprint


FIXED = dict(re_pc=300., rs_pc=500., q=.7, Q=.8, alpha=2., beta=3.,
             gamma=.5, beta_z=-.3, inclination=1.1)


def make_model(**kwargs):
    options = dict(data=dict(x_pc=[0., 100., -200.], y_pc=[0., -50., 150.],
                            vlos_kms=[1., -3., 5.], e_vlos_kms=[0., 1., 2.]),
                   prior=pd.DataFrame(dict(lower=[-1.5, -20.], upper=[-.5, 20.]),
                                      index=["log10_rhos_Msunpc3", "vmem_kms"]),
                   dsph_model=AxisymmetricDSphModel(16, 16, 16), fixed_params=FIXED)
    options.update(kwargs)
    return AxisymmetricDSphEstimationModel(**options)


def test_gaussian_likelihood_and_public_api():
    import jeanspy
    from jeanspy import model
    m = make_model()
    p = np.array([-1., 2.])
    physical = {**FIXED, "rhos_Msunpc3": .1, "vmem_kms": 2.}
    s2 = m.dsph_model.sigmalos2(m.data.x_pc, m.data.y_pc, params=physical)
    expected = norm.logpdf(m.data.vlos_kms, loc=2., scale=np.sqrt(s2+m.data.e_vlos_kms**2))
    np.testing.assert_allclose(m.lnlikelihoods(p), expected, rtol=1e-13)
    np.testing.assert_allclose(m.lnposterior(p), [sum(expected), sum(expected), 0., 0.])
    assert model.AxisymmetricDSphModel is AxisymmetricDSphModel
    assert model.AxisymmetricDSphEstimationModel is AxisymmetricDSphEstimationModel
    for name in jeanspy.__all__:
        assert getattr(jeanspy, name) is not None


@pytest.mark.parametrize("change", [dict(x_pc=[]), dict(y_pc=[1., 2.]),
    dict(vlos_kms=[np.nan, 0., 1.]), dict(e_vlos_kms=[0., -1., 0.]), dict(x_pc=[[0., 1., 2.]])])
def test_observation_validation_is_atomic(change):
    m = make_model()
    before = fingerprint(m)
    data = m.data.as_kwargs()
    data.update(change)
    with pytest.raises(ValueError):
        m.reset_data(data)
    assert fingerprint(m) == before


def test_data_are_detached_and_temperature_tracks_reset():
    m = make_model()
    before = fingerprint(m)
    data = m.data.as_kwargs()
    data["y_pc"][0] = 100.
    assert fingerprint(m) == before
    np.testing.assert_allclose(m.inverse_temparature, 1/np.log(3))
    np.testing.assert_allclose(m.lnposterior_wbic([-1., 0.])[1],
                               m.lnlikelihood([-1., 0.])/np.log(3))
    m.reset_data({name: value[:1] for name, value in data.items()})
    with pytest.raises(ValueError, match="at least two"):
        m.lnposterior_wbic([-1., 0.])


def test_invalid_proposals_reject_without_poisoning_subsequent_calls():
    m = make_model()
    good = m.lnposterior([-1., 0.])
    for p in ([np.nan, 0.], [1e6, 0.], [-1., np.inf], [-4., 0.]):
        assert m.lnposterior(p)[0] == -np.inf
        np.testing.assert_equal(m.lnposterior([-1., 0.]), good)
    impossible = make_model(fixed_params={**FIXED, "beta_z": .99})
    assert impossible.lnlikelihood([-1., 0.]) == -np.inf
    with pytest.raises(RuntimeError, match="feasible prior draws"):
        impossible.sample(1, max_attempts=2, rng=1)


def test_coordinate_transforms_and_photometry_are_explicit():
    fixed = {key: value for key, value in FIXED.items() if key not in {"re_pc", "q", "inclination", "beta_z"}}
    fixed.update(rhos_Msunpc3=.1, vmem_kms=0., q_projected=.8)
    prior = pd.DataFrame(dict(lower=[2., 0., 0.], upper=[3., .7, .5]),
                         index=["log10_re_pc", "cos_inclination", "bfunc_beta_z"])
    m = make_model(prior=prior, fixed_params=fixed, photometry_prior=PhotometryPriorModel(2.4, .2))
    p = [2.4, .2, .1]
    values = m.convert_params(p)
    np.testing.assert_allclose([values["re_pc"], values["inclination"], values["beta_z"]],
                               [10**2.4, np.arccos(.2), 1-10**.1])
    assert m.lnpriors(p)[-1] == pytest.approx(norm.logpdf(2.4, 2.4, .2))
    draws = m.sample(3, rng=18)
    np.testing.assert_equal(draws, m.sample(3, rng=18))
    assert all(np.isfinite(m.lnposterior(row)[0]) for row in draws)
    # Incompatible projected flattening is rejected even within the flat bounds.
    m.prior.data.loc["cos_inclination", "upper"] = .99
    assert m.lnposterior([2.4, .95, .1])[0] == -np.inf


@pytest.mark.parametrize("change,match", [
    (dict(fixed_params={**FIXED, "vmem_kms": 0.}), "disjoint"),
    (dict(fixed_params={**FIXED, "q_projected": .8}), "exactly one"),
    (dict(fixed_params={**FIXED, "typo": 1.}), "Unknown"),
    (dict(fixed_params={**FIXED, "Q": np.nan}), "finite scalar"),
    (dict(photometry_prior=PhotometryPriorModel(2.4, .1)), "log10_re_pc"),
])
def test_bad_schema_fails_before_sampling(change, match):
    with pytest.raises(ValueError, match=match):
        make_model(**change)


def test_sampling_identity_excludes_evaluated_coordinates():
    m = make_model()
    before = fingerprint(m)
    m.lnposterior([-1.1, 2.])
    assert fingerprint(m) == before == fingerprint(make_model())
    for changed in (make_model(dsph_model=AxisymmetricDSphModel(24, 16, 16)),
                    make_model(fixed_params={**FIXED, "Q": .9})):
        assert fingerprint(changed) != before
    m.prior.data = m.prior.data.iloc[::-1].copy()
    assert fingerprint(m) != before


def test_classical_sampler_resume_and_changed_sky_coordinate_rejection(tmp_path):
    m = make_model()
    prefix = str(tmp_path / "axis_")
    first = Sampler(m, m.sample, nwalkers=6, prefix=prefix)
    first.run_mcmc(3, 1, enable_convergence_check=False)
    m2 = make_model()
    resumed = Sampler(m2, m2.sample, nwalkers=6, prefix=prefix)
    resumed.run_mcmc(2, 1, enable_convergence_check=False)
    assert resumed.get_chain().shape == (5, 6, 2)
    assert np.isfinite(resumed.get_log_prob()).all()
    frame = resumed.get_dataframe(discard=1)
    assert list(frame.columns) == m.p_names_lnprob+["lnprob"]
    assert len(frame) == 24
    data = m2.data.as_kwargs()
    data["y_pc"][1] += 1.
    m2.reset_data(data)
    with pytest.raises(ValueError, match="identity"):
        resumed.run_mcmc(1, 1)
    assert resumed.backend.iteration == 5
    with pytest.raises(ValueError, match="identity"):
        Sampler(m2, m2.sample, nwalkers=6, prefix=prefix)


def test_mock_velocities_and_zero_sized_draws():
    m = make_model()
    np.testing.assert_equal(m.sample_data([-1., 0.], rng=11), m.sample_data([-1., 0.], rng=11))
    assert m.sample(0).shape == (0, m.ndim)
    with pytest.raises(ValueError, match="shape"):
        m.convert_params([-1.])
