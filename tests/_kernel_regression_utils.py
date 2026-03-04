import jax.numpy as jnp
import numpy as np

from jeanspy.model_numpyro import BaesAnisotropyModel, ConstantAnisotropyModel


def assert_baes_constant_large_u_consistency(
    *,
    beta_values: tuple[float, ...],
    u_max: float = 1e5,
    n_u: int = 260,
    r_value: float = 1.0,
    n_kernel: int = 512,
    rtol: float = 3e-4,
    atol: float = 3e-6,
) -> None:
    """Assert BAES(beta_0=beta_inf) matches constant anisotropy in large-u regime."""
    baes = BaesAnisotropyModel()
    const = ConstantAnisotropyModel()

    u = jnp.asarray(np.geomspace(1.0 + 1e-4, float(u_max), int(n_u)), dtype=jnp.float32)[None, :]
    r = jnp.asarray([float(r_value)], dtype=jnp.float32)[:, None]

    for beta_const in beta_values:
        params_baes = {
            "beta_0": float(beta_const),
            "beta_inf": float(beta_const),
            "r_a": 1.0,
            "eta": 1.0,
        }
        params_const = {"beta_ani": float(beta_const)}

        k_baes = np.asarray(baes.kernel(u, r, params=params_baes, n_kernel=int(n_kernel))).reshape(-1)
        k_const = np.asarray(const.kernel(u, r, params=params_const)).reshape(-1)

        assert np.isfinite(k_baes).all(), f"BAES non-finite at beta={beta_const}"
        assert np.isfinite(k_const).all(), f"CONST non-finite at beta={beta_const}"
        np.testing.assert_allclose(
            k_baes,
            k_const,
            rtol=float(rtol),
            atol=float(atol),
            err_msg=f"large-u mismatch at beta={beta_const}",
        )
