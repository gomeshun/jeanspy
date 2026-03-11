import unittest
from unittest import mock

import numpy as np
import jax.numpy as jnp

import jeanspy.model_numpyro as model_numpyro_mod
from jeanspy.model_numpyro import ConstantAnisotropyModel, DSphModel, NFWModel, PlummerModel


class TestRuntimeConfig(unittest.TestCase):
    def test_default_sigmalos2_n_u_gpu_x32(self):
        with mock.patch.dict(model_numpyro_mod.os.environ, {}, clear=False):
            model_numpyro_mod.os.environ.pop("JEANSPY_SIGMALOS2_N_U", None)
            with mock.patch.object(model_numpyro_mod, "_prefers_gpu_x32", return_value=True):
                self.assertEqual(model_numpyro_mod._default_sigmalos2_n_u(), 1024)

    def test_default_sigmalos2_n_u_respects_env_override(self):
        with mock.patch.dict(model_numpyro_mod.os.environ, {"JEANSPY_SIGMALOS2_N_U": "777"}, clear=False):
            with mock.patch.object(model_numpyro_mod, "_prefers_gpu_x32", return_value=True):
                self.assertEqual(model_numpyro_mod._default_sigmalos2_n_u(), 777)

    def test_configure_runtime_round_trip(self):
        original = model_numpyro_mod.get_runtime_config()
        try:
            updated = model_numpyro_mod.configure_runtime(
                hyp2f1_backend="scipy",
                sigmalos2_backend="kernel",
                sigmalos2_jit=False,
                sigmalos2_n_u=4097,
                sigmalos2_n_r=640,
                sigmalos2_u_max=4096.0,
                constant_kernel_n_quad=24,
            )
            self.assertEqual(updated["hyp2f1_backend"], "scipy")
            self.assertEqual(updated["sigmalos2_backend_default"], "kernel")
            self.assertEqual(updated["sigmalos2_jit_default"], "false")
            self.assertEqual(updated["sigmalos2_n_u_default"], 4097)
            self.assertEqual(updated["sigmalos2_n_r_default"], 640)
            self.assertEqual(updated["sigmalos2_u_max_default"], 4096.0)
            self.assertEqual(updated["constant_kernel_n_quad"], 24)
        finally:
            model_numpyro_mod.configure_runtime(
                hyp2f1_backend=original["hyp2f1_backend"],
                hyp2f1_jax_method=original["hyp2f1_jax_method"],
                hyp2f1_jax_quad_rule=original["hyp2f1_jax_quad_rule"],
                sigmalos2_backend=original["sigmalos2_backend_default"],
                sigmalos2_jit=original["sigmalos2_jit_default"],
                sigmalos2_n_u=original["sigmalos2_n_u_default"],
                sigmalos2_n_r=original["sigmalos2_n_r_default"],
                sigmalos2_u_max=original["sigmalos2_u_max_default"],
                constant_kernel_n_quad=original["constant_kernel_n_quad"],
                jax_enable_x64=original["jax_enable_x64"],
            )

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