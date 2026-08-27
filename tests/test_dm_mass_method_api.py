"""Regression contract for NumPyro DM enclosed-mass method selection.

This follow-up to PR #29 intentionally describes the pre-v0.1 public API we
want to preserve going forward: a method-style ``dm_mass_method`` option and a
capability-driven ``auto`` policy.
"""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from jeanspy.model_numpyro import (
    ConstantAnisotropyModel,
    DSphModel,
    DMModel,
    NFWModel,
    PlummerModel,
    ZhaoModel,
)


def _zhao_dsph() -> DSphModel:
    return DSphModel(
        submodels={
            "StellarModel": PlummerModel(),
            "DMModel": ZhaoModel(),
            "AnisotropyModel": ConstantAnisotropyModel(),
        }
    )


def _zhao_params() -> dict[str, float]:
    return {
        "re_pc": 220.0,
        "rs_pc": 900.0,
        "rhos_Msunpc3": 8e-3,
        "a": 1.2,
        "b": 4.2,
        "g": 0.6,
        "r_t_pc": 8000.0,
        "beta_ani": 0.2,
        "vmem_kms": 0.0,
    }


def test_public_jeans_api_uses_dm_mass_method_not_boolean_flag():
    for method_name in ("sigmalos2", "sigmalos2_kernel", "sigmalos2_abel"):
        signature = inspect.signature(getattr(DSphModel, method_name))
        assert "dm_mass_method" in signature.parameters
        assert signature.parameters["dm_mass_method"].default == "auto"
        assert "use_analytic_dm" not in signature.parameters


def test_auto_selection_is_capability_driven():
    assert hasattr(DMModel, "analytic_enclosed_mass_autodiff_safe")
    assert DMModel.analytic_enclosed_mass_autodiff_safe is False
    assert NFWModel.analytic_enclosed_mass_autodiff_safe is True
    assert ZhaoModel.analytic_enclosed_mass_autodiff_safe is False


def test_nfw_auto_matches_analytic_and_zhao_auto_matches_numeric():
    r = jnp.asarray(np.geomspace(1.0, 3000.0, 32), dtype=jnp.float32)

    nfw = NFWModel()
    nfw_params = {
        "rs_pc": 900.0,
        "rhos_Msunpc3": 8e-3,
        "r_t_pc": 8000.0,
    }
    np.testing.assert_allclose(
        np.asarray(nfw.enclosed_mass(r, method="auto", params=nfw_params)),
        np.asarray(nfw.enclosed_mass(r, method="analytic", params=nfw_params)),
        rtol=0.0,
        atol=0.0,
    )

    zhao = ZhaoModel()
    zhao_params = {
        "rs_pc": 900.0,
        "rhos_Msunpc3": 8e-3,
        "a": 1.2,
        "b": 4.2,
        "g": 0.6,
        "r_t_pc": 8000.0,
    }
    np.testing.assert_allclose(
        np.asarray(zhao.enclosed_mass(r, method="auto", params=zhao_params)),
        np.asarray(zhao.enclosed_mass(r, method="numeric", params=zhao_params)),
        rtol=0.0,
        atol=0.0,
    )


def test_invalid_dm_mass_method_is_rejected_consistently():
    dsph = _zhao_dsph()
    params = _zhao_params()
    R = jnp.asarray([20.0, 100.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="dm_mass_method"):
        dsph.sigmalos2(
            R,
            params=params,
            backend="kernel",
            jit=False,
            n_u=32,
            u_max=300.0,
            dm_mass_method="not-a-method",
        )
