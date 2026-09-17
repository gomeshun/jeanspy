"""Physical meaning, coordinate measures and observation-precision contracts."""
from dataclasses import replace
from contextlib import contextmanager
import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from jeanspy import model
from jeanspy.parameters import SamplingParameter
from jeanspy._sampling_identity import fingerprint


@contextmanager
def double_precision(jax):
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def observations(dtype=np.float64):
    return pd.DataFrame(dict(R_pc=[100.123456789, 200.987654321],
                             vlos_kms=[-1.123456789, 2.987654321],
                             e_vlos_kms=[.123456789, .987654321]), dtype=dtype)


def fit(config, data=None, **kwargs):
    return model.get_default_estimation_model(
        observations() if data is None else data, 2.3, .1, config=config, **kwargs)


def test_projected_exponential_scale_and_half_light_radius():
    tracer = model.ProjectedExponentialModel(r_exp_pc=100.)
    assert tracer.re_pc == pytest.approx(167.834699001666)
    assert tracer.half_light_radius() == tracer.re_pc
    assert tracer.cdf_R(tracer.re_pc) == pytest.approx(.5, abs=2e-15)
    for density, measure in [(tracer.density_2d, lambda r: 2*np.pi*r),
                             (tracer.density_3d, lambda r: 4*np.pi*r*r)]:
        integral = quad(lambda r: measure(r)*density(r), 0., np.inf, epsabs=1e-10)[0]
        assert integral == pytest.approx(1., rel=1e-9)
    np.testing.assert_allclose(np.exp(tracer.logdensity_2d(np.array([10., 100.]))),
                               tracer.density_2d(np.array([10., 100.])))
    tracer.update(r_exp_pc=200.)
    assert tracer.re_pc == pytest.approx(335.669398003332)
    with pytest.raises(AttributeError):
        tracer.re_pc = 100.
    with pytest.raises(ValueError, match="Unknown parameter 're_pc'"):
        model.ProjectedExponentialModel(re_pc=100.)


def test_removed_update_target_is_rejected_before_mutation():
    tracer = model.PlummerModel(re_pc=100.)
    assert "target" not in inspect.signature(tracer.update).parameters
    with pytest.raises(ValueError, match="Unknown parameter 'target'"):
        tracer.update(re_pc=200., target="DMModel")
    assert tracer.params.re_pc == 100.


@pytest.mark.parametrize("kind", ["NFWModel", "ZhaoModel"])
def test_truncated_density_matches_mass_and_jax_gradients(kind):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jeanspy import model_jax
    with double_precision(jax):
        params = dict(rs_pc=100., rhos_Msunpc3=.1, r_t_pc=500.)
        if kind == "ZhaoModel":
            params.update(alpha=1.2, beta=3.4, gamma=.6)
        halo = getattr(model, kind)(**params)
        functional = getattr(model_jax, kind)()
        radii = np.array([30., 200., 500., 501., 1000.])
        rho = halo.mass_density_3d(radii)
        np.testing.assert_array_equal(rho[-2:], 0.)
        assert rho[2] > 0.  # The boundary is part of the halo.
        calculate = jax.jit(lambda p: functional.mass_density_3d(jnp.asarray(radii), params=p))
        np.testing.assert_allclose(calculate(params), rho, rtol=1e-12)
        for r in (30., 200., 501., 1000.):
            derivative = jax.grad(lambda radius: functional.enclosed_mass(radius, params=params))(r)
            assert float(derivative) == pytest.approx(4*np.pi*r*r*halo.mass_density_3d(r), rel=1e-9)
        grad = jax.grad(lambda rho_s: calculate({**params, "rhos_Msunpc3": rho_s}).sum())(.1)
        assert float(grad) == pytest.approx(rho.sum()/.1, rel=1e-12)
        assert halo.enclosed_mass(1000.) == halo.enclosed_mass(500.)
        uncut = getattr(model, kind)(**{**params, "r_t_pc": np.inf})
        assert uncut.mass_density_3d(1000.) > 0.
        with pytest.raises(ValueError):
            halo.mass_density_3d(-1.)
        assert np.isnan(functional.mass_density_3d(-1., params=params))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_observations_keep_input_precision(classical_prior_config, dtype, shared):
    data = observations(dtype)
    target = fit(classical_prior_config, data)
    if shared:
        target.load_data(data, shared=True)
    try:
        assert target.dtype == np.dtype(dtype)
        for field in data:
            np.testing.assert_array_equal(target.data[field], data[field].to_numpy())
            assert target.data[field].dtype == dtype
        replacement = data + dtype(.01)
        target.reset_data(replacement)
        for field in data:
            np.testing.assert_array_equal(target.data[field], replacement[field].to_numpy())
        if shared:
            assert target.buffer_size == len(data)*np.dtype(dtype).itemsize
    finally:
        if shared:
            target.release_shared_memory()


def test_requested_precision_and_shared_dtype_change_are_explicit(classical_prior_config):
    data = observations()
    reduced = fit(classical_prior_config, data, dtype=np.float32)
    precise = fit(classical_prior_config, data, dtype=np.float64)
    assert fingerprint(precise) != fingerprint(reduced)
    np.testing.assert_array_equal(reduced.data.R_pc, data.R_pc.to_numpy(dtype=np.float32))
    np.testing.assert_array_equal(precise.data.R_pc, data.R_pc.to_numpy())
    for invalid in (np.int64, np.complex128, "object"):
        with pytest.raises(ValueError, match="floating dtype"):
            fit(classical_prior_config, dtype=invalid)
    target = fit(classical_prior_config, observations(np.float32))
    target.load_data(observations(np.float32), shared=True)
    try:
        before = fingerprint(target)
        with pytest.raises(ValueError, match="Cannot change shared observation dtype"):
            target.reset_data(data)
        assert fingerprint(target) == before
    finally:
        target.release_shared_memory()


def test_mixed_column_precision_uses_lossless_common_dtype(classical_prior_config):
    data = observations().astype({"R_pc": np.float32})
    target = fit(classical_prior_config, data)
    assert target.dtype == np.dtype(np.float64)
    for field in data:
        np.testing.assert_array_equal(target.data[field], data[field].to_numpy())


def test_names_do_not_select_transform_and_prior_stays_in_sample_coordinates(classical_prior_config):
    target = fit(classical_prior_config)
    values = np.array([0., 2.3, 3., -2., 4., .1])
    expected = target.convert_params(values)
    original_prior = target.lnpriors(values)
    # Arbitrary labels, including misleading prefixes, have exactly the declared meaning.
    names = ["log10_velocity", "size", "halo_size", "amplitude", "edge", "anisotropy"]
    target["FlatPriorModel"].data.index = names
    target.parameter_specs = tuple(replace(spec, sample_name=name)
                                   for name, spec in zip(names, target.parameter_specs))
    pd.testing.assert_series_equal(target.convert_params(values), expected)
    np.testing.assert_allclose(target.lnpriors(values), original_prior)
    assert expected.beta_ani == pytest.approx(1-10**.1)
    assert target.lnpriors(values)[1] == pytest.approx(norm.logpdf(2.3, 2.3, .1))
    before = fingerprint(target)
    target.parameter_specs = (*target.parameter_specs[:-1],
                              replace(target.parameter_specs[-1], transform="identity"))
    assert fingerprint(target) != before
    assert target.convert_params(values).beta_ani == .1


def test_exponential_photometry_prior_is_on_half_light_radius(classical_prior_config):
    target = fit(classical_prior_config)
    submodels = dict(target["DSphModel"].submodels)
    submodels["StellarModel"] = model.ProjectedExponentialModel()
    specs = list(target.parameter_specs)
    specs[1] = SamplingParameter("log10_scale", "r_exp_pc", "pow10")
    prior = classical_prior_config.rename(index={"log10_re_pc": "log10_scale"})
    target = model.SimpleDSphEstimationModel(
        args_load_data=[observations()], parameter_specs=specs,
        submodels={"DSphModel": model.DSphModel(submodels=submodels),
                   "FlatPriorModel": model.FlatPriorModel(prior),
                   "PhotometryPriorModel": model.PhotometryPriorModel(2.3, .1)})
    p = np.array([0., 2.3-np.log10(1.67834699001666), 3., -2., 4., 0.])
    assert target.lnpriors(p)[1] == pytest.approx(norm.logpdf(2.3, 2.3, .1))
    target.update(target.convert_params(p))
    assert target["DSphModel"]["StellarModel"].re_pc == pytest.approx(10**2.3)
    draws = target.sample(10)
    assert all(np.isfinite(target.lnpriors(row)).all() for row in draws)
