"""Regression tests for the regularized kernel outer integration coordinate."""

import numpy as np
import jax.numpy as jnp
import pytest

from jeanspy.model_numpyro import (
    BaesAnisotropyModel,
    ConstantAnisotropyModel,
    DSphModel,
    NFWModel,
    OsipkovMerrittModel,
    PlummerModel,
    get_runtime_config,
)


def _dsph(anisotropy):
    return DSphModel(
        submodels={
            "StellarModel": PlummerModel(),
            "DMModel": NFWModel(),
            "AnisotropyModel": anisotropy,
        }
    )


def _max_rel(value, reference) -> float:
    value = np.asarray(value, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    floor = max(float(np.max(np.abs(reference))) * 1e-8, 1e-10)
    return float(
        np.max(np.abs(value - reference) / np.maximum(np.abs(reference), floor))
    )


def _common_params():
    return {
        "re_pc": 220.0,
        "rs_pc": 1100.0,
        "rhos_Msunpc3": 7.5e-3,
        "r_t_pc": 8800.0,
        "vmem_kms": 0.0,
    }


def test_sqrtlog_is_reported_as_kernel_outer_default():
    config = get_runtime_config()
    assert config["sigmalos2_kernel_outer_transform_default"] == "sqrtlog"


def test_invalid_kernel_outer_transform_is_rejected():
    dsph = _dsph(ConstantAnisotropyModel())
    params = {**_common_params(), "beta_ani": 0.0}
    with pytest.raises(ValueError, match="kernel_outer_transform"):
        dsph.sigmalos2(
            jnp.asarray([100.0]),
            params=params,
            backend="kernel",
            kernel_outer_transform="not-a-transform",
            jit=False,
        )


def test_sqrtlog_resolves_extreme_tangential_endpoint_far_better_than_log():
    """The MCMC-adversarial beta=-5 case exposes the old endpoint error."""
    dsph = _dsph(BaesAnisotropyModel())
    params = {
        **_common_params(),
        "beta_0": -5.0,
        "beta_inf": -5.0,
        "r_a": 220.0,
        "eta": 2.0,
    }
    R = jnp.asarray(np.geomspace(1.1, 2200.0, 24), dtype=jnp.float32)

    reference = dsph.sigmalos2(
        R,
        params=params,
        backend="kernel",
        n_u=512,
        n_kernel=64,
        u_max=2.0e4,
        kernel_outer_transform="sqrtlog",
        dm_mass_method="analytic",
        jit=False,
    )
    sqrtlog = dsph.sigmalos2(
        R,
        params=params,
        backend="kernel",
        n_u=64,
        n_kernel=32,
        u_max=2.0e4,
        kernel_outer_transform="sqrtlog",
        dm_mass_method="analytic",
        jit=False,
    )
    legacy_log = dsph.sigmalos2(
        R,
        params=params,
        backend="kernel",
        n_u=64,
        n_kernel=32,
        u_max=2.0e4,
        kernel_outer_transform="log",
        dm_mass_method="analytic",
        jit=False,
    )

    sqrtlog_err = _max_rel(sqrtlog, reference)
    log_err = _max_rel(legacy_log, reference)

    assert np.isfinite(np.asarray(sqrtlog)).all()
    assert sqrtlog_err < 5e-3
    assert sqrtlog_err < 0.1 * log_err


@pytest.mark.parametrize(
    ("anisotropy", "ani_params"),
    [
        (ConstantAnisotropyModel(), {"beta_ani": 0.7}),
        (OsipkovMerrittModel(), {"r_a": 300.0}),
    ],
)
def test_sqrtlog_generalizes_beyond_baes(anisotropy, ani_params):
    """The regularized outer coordinate is not specific to the BAES kernel."""
    dsph = _dsph(anisotropy)
    params = {**_common_params(), **ani_params}
    R = jnp.asarray(np.geomspace(5.0, 1500.0, 18), dtype=jnp.float32)

    sqrtlog = dsph.sigmalos2(
        R,
        params=params,
        backend="kernel",
        n_u=128,
        n_kernel=64,
        u_max=2.0e4,
        kernel_outer_transform="sqrtlog",
        dm_mass_method="analytic",
        jit=False,
    )
    legacy_log_ref = dsph.sigmalos2(
        R,
        params=params,
        backend="kernel",
        n_u=1024,
        n_kernel=64,
        u_max=2.0e4,
        kernel_outer_transform="log",
        dm_mass_method="analytic",
        jit=False,
    )

    assert np.isfinite(np.asarray(sqrtlog)).all()
    assert np.isfinite(np.asarray(legacy_log_ref)).all()
    assert _max_rel(sqrtlog, legacy_log_ref) < 8e-3
