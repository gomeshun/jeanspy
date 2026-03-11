import unittest
from unittest import mock

import numpy as np
import jax.numpy as jnp

import jeanspy.model_numpyro as model_numpyro_mod
from jeanspy.model_numpyro import ConstantAnisotropyModel, DSphModel, NFWModel, PlummerModel


class TestRuntimeConfig(unittest.TestCase):
    def test_default_sigmalos2_n_u_gpu_x32(self):
        with mock.patch.object(model_numpyro_mod, "_prefers_gpu_x32", return_value=True):
            self.assertEqual(model_numpyro_mod._default_sigmalos2_n_u(), 1024)

    def test_default_sigmalos2_n_u_has_no_env_override(self):
        with mock.patch.dict(model_numpyro_mod.os.environ, {"JEANSPY_SIGMALOS2_N_U": "777"}, clear=False):
            with mock.patch.object(model_numpyro_mod, "_prefers_gpu_x32", return_value=True):
                self.assertEqual(model_numpyro_mod._default_sigmalos2_n_u(), 1024)

    def test_configure_runtime_updates_jax_precision_only(self):
        original = bool(model_numpyro_mod.jax.config.read("jax_enable_x64"))
        try:
            updated = model_numpyro_mod.configure_runtime(jax_enable_x64=not original)
            self.assertEqual(updated["jax_enable_x64"], (not original))
        finally:
            model_numpyro_mod.configure_runtime(jax_enable_x64=original)

    def test_configure_runtime_rejects_legacy_numerical_options(self):
        with self.assertRaises(TypeError):
            model_numpyro_mod.configure_runtime(sigmalos2_n_u=1024)

    def test_sigmalos2_jit_matches_eager(self):
        params = {
            "re_pc": 200.0,
            "rs_pc": 1200.0,
            "rhos_Msunpc3": 1e-2,
            "r_t_pc": 8000.0,
            "beta_ani": 0.2,
            "vmem_kms": 0.0,
        }
        dsph = DSphModel(
            submodels={
                "StellarModel": PlummerModel(),
                "DMModel": NFWModel(),
                "AnisotropyModel": ConstantAnisotropyModel(),
            }
        )
        radii = jnp.geomspace(10.0, 1000.0, 12)
        params_jax = {key: jnp.asarray(value, dtype=jnp.float32) for key, value in params.items()}

        eager = np.asarray(dsph.sigmalos2(radii, params=params_jax, backend="kernel", jit=False, n_u=256, u_max=2000.0))
        compiled = np.asarray(dsph.sigmalos2(radii, params=params_jax, backend="kernel", jit=True, n_u=256, u_max=2000.0))

        np.testing.assert_allclose(eager, compiled, rtol=2e-6, atol=5e-8)


if __name__ == "__main__":
    unittest.main()