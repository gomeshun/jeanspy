"""Validation tests for the eta=2 Baes--van Hese analytic kernel."""

import numpy as np
import jax.numpy as jnp

from jeanspy.baes_eta2 import (
    BaesEta2AnisotropyModel,
    baes_eta2_kernel_appell_reference,
)
from jeanspy.model_numpyro import (
    BaesAnisotropyModel,
    ConstantAnisotropyModel,
    DSphModel,
    NFWModel,
    OsipkovMerrittModel,
    PlummerModel,
)


def _broadcast_grid(u_values, r_values):
    u = jnp.asarray(np.asarray(u_values, dtype=np.float64))[None, :]
    R = jnp.asarray(np.asarray(r_values, dtype=np.float64))[:, None]
    return u, R


def test_eta2_appell_reference_matches_generic_baes_kernel():
    """Exact Appell-F1 expression agrees with the existing numerical kernel."""
    generic = BaesAnisotropyModel()
    u, R = _broadcast_grid([1.001, 1.03, 1.2, 2.0, 5.0, 10.0], [0.7, 2.3])

    cases = [
        {"beta_0": -0.5, "beta_inf": 0.7, "r_a": 1.4},
        {"beta_0": -2.0, "beta_inf": 0.3, "r_a": 0.8},
        {"beta_0": 0.1, "beta_inf": 0.8, "r_a": 3.0},
    ]

    for case in cases:
        params_generic = {**case, "eta": 2.0}
        k_numeric = np.asarray(
            generic.kernel(u, R, params=params_generic, n_kernel=512),
            dtype=np.float64,
        )
        k_appell = baes_eta2_kernel_appell_reference(
            np.broadcast_to(np.asarray(u), k_numeric.shape),
            np.broadcast_to(np.asarray(R), k_numeric.shape),
            case["beta_0"],
            case["beta_inf"],
            case["r_a"],
            dps=32,
        )

        assert np.isfinite(k_numeric).all()
        assert np.isfinite(k_appell).all()
        np.testing.assert_allclose(k_appell, k_numeric, rtol=3e-4, atol=2e-7)


def test_eta2_jax_evaluator_matches_appell_reference():
    """JAX evaluator of the analytic reduction matches high-precision Appell F1."""
    model = BaesEta2AnisotropyModel()
    u, R = _broadcast_grid([1.001, 1.02, 1.1, 1.5, 3.0, 8.0], [0.6, 1.7])
    params = {"beta_0": -1.2, "beta_inf": 0.65, "r_a": 1.5}

    k_jax = np.asarray(model.kernel(u, R, params=params, n_kernel=128), dtype=np.float64)
    k_appell = baes_eta2_kernel_appell_reference(
        np.broadcast_to(np.asarray(u), k_jax.shape),
        np.broadcast_to(np.asarray(R), k_jax.shape),
        params["beta_0"],
        params["beta_inf"],
        params["r_a"],
        dps=32,
    )

    assert np.isfinite(k_jax).all()
    assert np.isfinite(k_appell).all()
    np.testing.assert_allclose(k_jax, k_appell, rtol=8e-4, atol=2e-6)


def test_eta2_kernel_reduces_to_constant_anisotropy():
    """beta_0=beta_inf reproduces the known constant-anisotropy kernel."""
    beta = -0.7
    eta2 = BaesEta2AnisotropyModel()
    const = ConstantAnisotropyModel()
    u, R = _broadcast_grid(np.geomspace(1.0 + 1e-4, 100.0, 120), [0.5, 2.0, 8.0])

    k_eta2 = np.asarray(
        eta2.kernel(
            u,
            R,
            params={"beta_0": beta, "beta_inf": beta, "r_a": 1.3},
            n_kernel=128,
        ),
        dtype=np.float64,
    )
    k_const = np.asarray(
        const.kernel(u, R, params={"beta_ani": beta}), dtype=np.float64
    )
    k_const = np.broadcast_to(k_const, k_eta2.shape)

    np.testing.assert_allclose(k_eta2, k_const, rtol=5e-4, atol=2e-6)


def test_eta2_kernel_reduces_to_osipkov_merritt():
    """(beta_0,beta_inf)=(0,1) reproduces the analytic OM kernel."""
    r_a = 1.8
    eta2 = BaesEta2AnisotropyModel()
    om = OsipkovMerrittModel()
    u, R = _broadcast_grid(np.geomspace(1.0 + 1e-4, 100.0, 120), [0.5, 1.5, 5.0])

    k_eta2 = np.asarray(
        eta2.kernel(
            u,
            R,
            params={"beta_0": 0.0, "beta_inf": 1.0, "r_a": r_a},
            n_kernel=160,
        ),
        dtype=np.float64,
    )
    k_om = np.asarray(om.kernel(u, R, params={"r_a": r_a}), dtype=np.float64)

    np.testing.assert_allclose(k_eta2, k_om, rtol=8e-4, atol=2e-6)


def test_eta2_sigmalos_kernel_matches_kernel_free_abel_solver():
    """Analytic-kernel LOS dispersion agrees with the independent Abel route."""
    params = {
        "re_pc": 220.0,
        "rs_pc": 1100.0,
        "rhos_Msunpc3": 7.5e-3,
        "r_t_pc": 9000.0,
        "beta_0": -0.5,
        "beta_inf": 0.65,
        "r_a": 300.0,
        "vmem_kms": 0.0,
    }
    dsph = DSphModel(
        submodels={
            "StellarModel": PlummerModel(),
            "DMModel": NFWModel(),
            "AnisotropyModel": BaesEta2AnisotropyModel(),
        }
    )
    R = jnp.asarray(np.geomspace(5.0, 900.0, 28), dtype=jnp.float32)

    s2_kernel = np.asarray(
        dsph.sigmalos2(
            R,
            params=params,
            backend="kernel",
            n_u=256,
            n_kernel=160,
            u_max=1600.0,
            use_analytic_dm=True,
            jit=False,
        ),
        dtype=np.float64,
    )
    s2_abel = np.asarray(
        dsph.sigmalos2(
            R,
            params=params,
            backend="abel",
            n_r=896,
            u_max=1600.0,
            r_min_factor=0.35,
            use_analytic_dm=True,
            jit=False,
        ),
        dtype=np.float64,
    )

    assert np.isfinite(s2_kernel).all()
    assert np.isfinite(s2_abel).all()
    assert (s2_kernel > 0.0).all()
    assert (s2_abel > 0.0).all()

    rel = np.abs(s2_kernel - s2_abel) / np.maximum(np.abs(s2_abel), 1e-10)
    assert float(np.max(rel)) < 6.0e-2
