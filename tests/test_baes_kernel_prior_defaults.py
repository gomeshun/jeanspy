import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import jeanspy.model_numpyro as mn
from jeanspy.model_numpyro import BaesAnisotropyModel, DSphModel, NFWModel, PlummerModel


def _params(*, eta=10.0, beta_0=-9.0, beta_inf=0.98, r_a=220.0, dtype=jnp.float32):
    raw = {
        "re_pc": 220.0,
        "rs_pc": 1100.0,
        "rhos_Msunpc3": 7.5e-3,
        "r_t_pc": 8800.0,
        "beta_0": beta_0,
        "beta_inf": beta_inf,
        "r_a": r_a,
        "eta": eta,
        "vmem_kms": 0.0,
    }
    return {key: jnp.asarray(value, dtype=dtype) for key, value in raw.items()}


def test_baes_default_kernel_order_is_prior_backed_32():
    assert mn.DEFAULT_BAES_KERNEL_N_QUAD == 32
    ani = BaesAnisotropyModel()
    params = _params()
    u = jnp.asarray([1.01, 1.2, 2.0, 10.0], dtype=jnp.float32)
    R = jnp.asarray(220.0, dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(ani.kernel(u, R, params=params)),
        np.asarray(ani.kernel(u, R, params=params, n_kernel=32)),
        rtol=0.0,
        atol=0.0,
    )


def test_baes_eta_above_ten_warns_but_eta_ten_does_not():
    ani = BaesAnisotropyModel()
    u = jnp.asarray([1.1, 2.0], dtype=jnp.float32)
    R = jnp.asarray(220.0, dtype=jnp.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ani.kernel(u, R, params=_params(eta=10.0))
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]

    with pytest.warns(RuntimeWarning, match=r"eta > 10"):
        ani.kernel(u, R, params=_params(eta=20.0))


def test_baes_float32_eta20_remains_numerically_finite():
    ani = BaesAnisotropyModel()
    params = _params(eta=20.0, beta_0=-5.0, beta_inf=0.98, r_a=0.22)
    u = jnp.asarray([1.0 + 1e-5, 1.1, 2.0, 10.0, 100.0, 1e3], dtype=jnp.float32)
    R = jnp.asarray(220.0, dtype=jnp.float32)
    with pytest.warns(RuntimeWarning):
        result = np.asarray(ani.kernel(u, R, params=params, n_kernel=64))
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) < 1e8


def test_nkernel32_matches_128_for_supported_eta_prior_projection():
    dsph = DSphModel(
        submodels={
            "StellarModel": PlummerModel(),
            "DMModel": NFWModel(),
            "AnisotropyModel": BaesAnisotropyModel(),
        }
    )
    params = _params(eta=10.0, beta_0=-9.0, beta_inf=0.98, r_a=220.0)
    R = jnp.asarray(np.geomspace(0.005, 10.0, 20) * 220.0, dtype=jnp.float32)
    common = dict(
        params=params,
        backend="kernel",
        jit=False,
        n_u=256,
        u_max=2e4,
        kernel_outer_transform="sqrtlog",
        dm_mass_method="analytic",
    )
    low = np.asarray(dsph.sigmalos2(R, n_kernel=32, **common), dtype=np.float64)
    high = np.asarray(dsph.sigmalos2(R, n_kernel=128, **common), dtype=np.float64)
    floor = max(float(np.max(np.abs(high))) * 1e-10, 1e-12)
    rel = np.max(np.abs(low - high) / np.maximum(np.abs(high), floor))
    assert rel < 2e-5
