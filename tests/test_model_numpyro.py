import unittest
import warnings
import time
from unittest.mock import patch

# IMPORTANT: must be set before importing JAX.
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import logging

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import AIES, MCMC, NUTS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import arviz as az
from tests._kernel_regression_utils import assert_baes_constant_large_u_consistency

from scipy.special import hyp2f1 as scipy_hyp2f1
import jeanspy.model_numpyro as model_numpyro_mod

from jeanspy.model_numpyro import (
    BaesAnisotropyModel,
    ConstantAnisotropyModel,
    DSphModel,
    Model,
    NFWModel,
    OsipkovMerrittModel,
    PlummerModel,
    ZhaoModel,
)


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _log_runtime():
    logger.info("JAX backend: %s", jax.default_backend())
    logger.info("JAX devices: %s", jax.devices())
    logger.info(
        "XLA_PYTHON_CLIENT_PREALLOCATE=%s XLA_PYTHON_CLIENT_MEM_FRACTION=%s XLA_PYTHON_CLIENT_ALLOCATOR=%s",
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
    )


class TestModelNumPyro(unittest.TestCase):
    def _make_sigmalos2_backend_cases(self):
        return [
            {
                "name": "constant",
                "anisotropy": ConstantAnisotropyModel(),
                "params": {
                    "re_pc": 220.0,
                    "rs_pc": 1100.0,
                    "rhos_Msunpc3": 7.5e-3,
                    "r_t_pc": 9000.0,
                    "beta_ani": 0.25,
                    "vmem_kms": 0.0,
                },
                "rtol_max": 3.5e-2,
            },
            {
                "name": "osipkov_merritt",
                "anisotropy": OsipkovMerrittModel(),
                "params": {
                    "re_pc": 220.0,
                    "rs_pc": 1100.0,
                    "rhos_Msunpc3": 7.5e-3,
                    "r_t_pc": 9000.0,
                    "r_a": 350.0,
                    "vmem_kms": 0.0,
                },
                "rtol_max": 5.0e-2,
            },
            {
                "name": "baes",
                "anisotropy": BaesAnisotropyModel(),
                "params": {
                    "re_pc": 220.0,
                    "rs_pc": 1100.0,
                    "rhos_Msunpc3": 7.5e-3,
                    "r_t_pc": 9000.0,
                    "beta_0": 0.0,
                    "beta_inf": 0.65,
                    "r_a": 300.0,
                    "eta": 2.2,
                    "vmem_kms": 0.0,
                },
                "rtol_max": 7.5e-2,
            },
        ]

    def _measure_jitted_runtime(self, fn, R_pc, repeats=5):
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = fn(R_pc)
            jax.block_until_ready(result)
            timings.append(time.perf_counter() - start)
        return float(np.median(np.asarray(timings, dtype=np.float64)))

    def _anisotropy_param_cases(self):
        """Parameter sets used for generic anisotropy consistency checks."""
        return {
            "ConstantAnisotropyModel": [
                {"beta_ani": -10.0},
                {"beta_ani": -2.0},
                {"beta_ani": 0.2},
                {"beta_ani": 1.0},
            ],
            "OsipkovMerrittModel": [
                {"r_a": 0.7},
                {"r_a": 1.5},
            ],
            "BaesAnisotropyModel": [
                {"beta_0": -10.0, "beta_inf": -10.0, "r_a": 1.2, "eta": 2.0},
                {"beta_0": -10.0, "beta_inf": 0.8, "r_a": 1.0, "eta": 2.0},
                {"beta_0": 0.0, "beta_inf": 1.0, "r_a": 1.5, "eta": 2.0},
                {"beta_0": 0.2, "beta_inf": 0.8, "r_a": 1.3, "eta": 3.0},
            ],
        }

    def _kernel_from_note_definition(self, model, R_values, u_values, params, n_quad=512):
        r"""Reference K(u) from the note definition used in model.py/model_numpyro.

        K(u_s) = f(Ru_s)/u_s * \int_1^{u_s} du [u/sqrt(u^2-1)]
                 * (1 - beta(Ru)/u^2) / f(Ru)
        """
        R_values = np.asarray(R_values, dtype=np.float64)
        u_values = np.asarray(u_values, dtype=np.float64)
        out = np.empty((R_values.shape[0], u_values.shape[0]), dtype=np.float64)

        for i, R in enumerate(R_values):
            for j, u_s in enumerate(u_values):
                y_max = np.sqrt(max(u_s * u_s - 1.0, 0.0))
                y = np.linspace(0.0, y_max, int(n_quad), dtype=np.float64)
                u_int = np.sqrt(1.0 + y * y)
                r_int = R * u_int

                beta_int = np.asarray(model.beta(jnp.asarray(r_int), params=params), dtype=np.float64)
                f_int = np.asarray(model.f(jnp.asarray(r_int), params=params), dtype=np.float64)
                integrand_y = (1.0 - beta_int / (u_int * u_int)) / f_int
                inner = np.trapezoid(integrand_y, y)

                f_s = float(np.asarray(model.f(jnp.asarray(R * u_s), params=params), dtype=np.float64))
                out[i, j] = (f_s / u_s) * inner

        return out

    def test_all_anisotropy_models_beta_f_kernel_consistency(self):
        """Check beta-f-kernel consistency for all AnisotropyModel subclasses.

        Consistency checks based on the note:
        1) d ln f / d ln r = 2 beta(r)
        2) kernel K(u) matches numerical reconstruction from beta and f.
        """
        subclasses = model_numpyro_mod.AnisotropyModel.__subclasses__()
        self.assertGreater(len(subclasses), 0)

        param_cases = self._anisotropy_param_cases()
        r_grid = np.geomspace(0.8, 2.5, 64).astype(np.float64)
        q = 1.0005
        u_grid = np.geomspace(1.0 + 1e-4, 4.0, 48).astype(np.float64)
        R_grid = np.array([0.9, 1.8], dtype=np.float64)

        for cls in subclasses:
            cls_name = cls.__name__
            self.assertIn(cls_name, param_cases, msg=f"Missing test parameter cases for {cls_name}")

            model = cls()
            for params in param_cases[cls_name]:
                # --- beta-f consistency: d ln f / d ln r = 2 beta ---
                r_plus = r_grid * q
                r_minus = r_grid / q

                f_plus = np.asarray(model.f(jnp.asarray(r_plus), params=params), dtype=np.float64)
                f_minus = np.asarray(model.f(jnp.asarray(r_minus), params=params), dtype=np.float64)
                beta_mid = np.asarray(model.beta(jnp.asarray(r_grid), params=params), dtype=np.float64)

                self.assertTrue(np.isfinite(f_plus).all(), msg=f"f_plus non-finite for {cls_name}, {params}")
                self.assertTrue(np.isfinite(f_minus).all(), msg=f"f_minus non-finite for {cls_name}, {params}")
                self.assertTrue(np.isfinite(beta_mid).all(), msg=f"beta non-finite for {cls_name}, {params}")

                dlogf_dlogr = (np.log(f_plus) - np.log(f_minus)) / (2.0 * np.log(q))
                np.testing.assert_allclose(
                    dlogf_dlogr,
                    2.0 * beta_mid,
                    rtol=4e-2,
                    atol=2e-3,
                    err_msg=f"beta-f mismatch for {cls_name}, params={params}",
                )

                # --- kernel consistency with note-based reconstruction ---
                k_model = np.asarray(
                    model.kernel(
                        jnp.asarray(u_grid)[None, :],
                        jnp.asarray(R_grid)[:, None],
                        params=params,
                    ),
                    dtype=np.float64,
                )
                k_ref = self._kernel_from_note_definition(model, R_grid, u_grid, params, n_quad=512)

                if k_model.shape != k_ref.shape:
                    k_model = np.broadcast_to(k_model, k_ref.shape)

                self.assertTrue(np.isfinite(k_model).all(), msg=f"kernel non-finite for {cls_name}, {params}")
                self.assertTrue(np.isfinite(k_ref).all(), msg=f"kernel ref non-finite for {cls_name}, {params}")
                np.testing.assert_allclose(
                    k_model,
                    k_ref,
                    rtol=3e-2,
                    atol=2e-4,
                    err_msg=f"kernel mismatch for {cls_name}, params={params}",
                )

    def test_constant_anisotropy_kernel_matches_scipy(self):
        """Kernel is the core of sigma_los; it must remain numerically stable.

        This test checks that our stabilized JAX implementation matches the original
        formula used in model.py (SciPy hyp2f1) over a wide range of u.
        """
        beta_ani = 0.2
        params = {"beta_ani": beta_ani}

        # u must be > 1. In float32 mode, the immediate endpoint vicinity
        # (u=1+1e-6) is quantization-sensitive; use practical model range.
        u_np = np.geomspace(1.0 + 1e-4, 3e3, 256).astype(np.float64)

        ani = ConstantAnisotropyModel()
        u = jnp.asarray(u_np)[None, :]
        R_dummy = jnp.asarray([10.0])[:, None]
        k_jax = np.asarray(ani.kernel(u, R_dummy, params=params)).reshape(-1)

        # SciPy reference (numerically stable form):
        # derived via the same exact transformation used in model_numpyro.
        u2 = u_np**2
        pref = np.sqrt(1.0 - 1.0 / u2)
        hyp_stable = scipy_hyp2f1(1.0, beta_ani, 1.5, 1.0 - 1.0 / u2)
        k_ref = pref * ((1.5 - beta_ani) * hyp_stable - 0.5)

        self.assertTrue(np.isfinite(k_jax).all())
        self.assertTrue(np.isfinite(k_ref).all())
        np.testing.assert_allclose(k_jax, k_ref, rtol=1e-3, atol=0.0)

        # Demonstrate equivalence with the original expression on a small-u regime
        # where SciPy's direct evaluation at z=1-u^2 is still well-behaved.
        u_small = np.geomspace(1.0 + 1e-6, 5.0, 64).astype(np.float64)
        u2s = u_small**2
        pref_s = np.sqrt(1.0 - 1.0 / u2s)
        hyp_orig = scipy_hyp2f1(1.0, 1.5 - beta_ani, 1.5, 1.0 - u2s)
        k_orig = pref_s * ((1.5 - beta_ani) * u2s * hyp_orig - 0.5)
        hyp_st = scipy_hyp2f1(1.0, beta_ani, 1.5, 1.0 - 1.0 / u2s)
        k_st = pref_s * ((1.5 - beta_ani) * hyp_st - 0.5)
        np.testing.assert_allclose(k_orig, k_st, rtol=1e-6, atol=0.0)

    def test_constant_anisotropy_kernel_matches_scipy_wide_beta_range(self):
        """Constant-anisotropy kernel stays accurate over the practical beta range."""
        beta_values = np.array([-10.0, -5.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9, 1.0])
        u_np = np.geomspace(1.0 + 1e-4, 5e2, 220).astype(np.float64)
        u = jnp.asarray(u_np)[None, :]
        R_dummy = jnp.asarray([100.0])[:, None]
        ani = ConstantAnisotropyModel()

        for beta_ani in beta_values:
            params = {"beta_ani": float(beta_ani)}
            k_jax = np.asarray(ani.kernel(u, R_dummy, params=params)).reshape(-1)

            u2 = u_np**2
            pref = np.sqrt(1.0 - 1.0 / u2)
            hyp_stable = scipy_hyp2f1(1.0, float(beta_ani), 1.5, 1.0 - 1.0 / u2)
            k_ref = pref * ((1.5 - float(beta_ani)) * hyp_stable - 0.5)

            self.assertTrue(np.isfinite(k_jax).all())
            self.assertTrue(np.isfinite(k_ref).all())
            np.testing.assert_allclose(
                k_jax,
                k_ref,
                rtol=1e-2,
                atol=1e-8,
                err_msg=f"failed at beta_ani={beta_ani}",
            )

    def test_constant_anisotropy_kernel_large_u_positive_beta_stable(self):
        """For beta>0, very large-u kernel should stay finite and follow asymptotic growth."""
        beta_ani = 0.5
        u_np = np.geomspace(1.0 + 1e-4, 1e16, 512).astype(np.float64)
        u = jnp.asarray(u_np, dtype=jnp.float32)
        R_dummy = jnp.asarray(1.0, dtype=jnp.float32)

        ani = ConstantAnisotropyModel()
        k = np.asarray(ani.kernel(u, R_dummy, params={"beta_ani": beta_ani}))

        self.assertTrue(np.isfinite(k).all())
        self.assertGreater(float(k[-1]), float(k[64]))

        # For beta=1/2: K(u) ~ log(2u) - 1/2 at large u.
        tail = u_np >= 1e8
        k_ref_tail = np.log(2.0 * u_np[tail]) - 0.5
        rel_tail = np.abs(k[tail] - k_ref_tail) / (np.abs(k_ref_tail) + 1e-300)
        self.assertLess(float(np.max(rel_tail)), 5e-2)

    def test_constant_anisotropy_kernel_large_u_positive_beta_stable_all_backends(self):
        """Large-u positive-beta stability should hold for both scipy and jax backends."""
        beta_ani = 0.5
        u_np = np.geomspace(1.0 + 1e-4, 1e16, 384).astype(np.float64)
        u = jnp.asarray(u_np, dtype=jnp.float32)
        r_dummy = jnp.asarray(1.0, dtype=jnp.float32)
        tail = u_np >= 1e8
        k_ref_tail = np.log(2.0 * u_np[tail]) - 0.5

        for backend in ("scipy", "jax"):
            with patch.object(model_numpyro_mod, "_HYP2F1_BACKEND", backend):
                ani = ConstantAnisotropyModel()
                k = np.asarray(ani.kernel(u, r_dummy, params={"beta_ani": beta_ani}))

            self.assertTrue(np.isfinite(k).all(), msg=f"non-finite kernel for backend={backend}")
            self.assertGreater(float(k[-1]), float(k[48]), msg=f"non-increasing tail for backend={backend}")

            rel_tail = np.abs(k[tail] - k_ref_tail) / (np.abs(k_ref_tail) + 1e-300)
            self.assertLess(
                float(np.max(rel_tail)),
                6e-2,
                msg=f"tail asymptotic mismatch for backend={backend}",
            )

    def test_constant_anisotropy_kernel_jax_matches_scipy_positive_beta(self):
        """JAX and SciPy backends should agree for beta>0 over wide u range."""
        u_np = np.geomspace(1.0 + 1e-4, 1e16, 1024).astype(np.float64)
        u = jnp.asarray(u_np, dtype=jnp.float32)
        r_dummy = jnp.asarray(100.0, dtype=jnp.float32)

        for beta_ani in (0.2, 0.5, 0.8, 1.0):
            with patch.object(model_numpyro_mod, "_HYP2F1_BACKEND", "scipy"):
                k_scipy = np.asarray(ConstantAnisotropyModel().kernel(u, r_dummy, params={"beta_ani": beta_ani}))

            with patch.object(model_numpyro_mod, "_HYP2F1_BACKEND", "jax"):
                k_jax = np.asarray(ConstantAnisotropyModel().kernel(u, r_dummy, params={"beta_ani": beta_ani}))

            self.assertTrue(np.isfinite(k_scipy).all(), msg=f"non-finite scipy kernel at beta={beta_ani}")
            self.assertTrue(np.isfinite(k_jax).all(), msg=f"non-finite jax kernel at beta={beta_ani}")

            rel = np.abs(k_jax - k_scipy) / (np.abs(k_scipy) + 1e-30)
            self.assertLess(
                float(np.max(rel)),
                2e-2,
                msg=f"jax/scipy mismatch too large at beta={beta_ani}",
            )

    def test_baes_kernel_reduces_to_constant_anisotropy(self):
        """BAES with beta_0=beta_inf reduces to the constant-anisotropy kernel."""
        beta_const = 0.2
        params_baes = {
            "beta_0": beta_const,
            "beta_inf": beta_const,
            "r_a": 300.0,
            "eta": 2.0,
        }
        params_const = {"beta_ani": beta_const}

        u = jnp.asarray(np.geomspace(1.0 + 1e-4, 300.0, 192), dtype=jnp.float32)[None, :]
        R = jnp.asarray([50.0, 200.0, 700.0], dtype=jnp.float32)[:, None]

        k_baes = np.asarray(BaesAnisotropyModel().kernel(u, R, params=params_baes, n_kernel=256))
        k_const = np.asarray(ConstantAnisotropyModel().kernel(u, R, params=params_const))
        k_const = np.broadcast_to(k_const, k_baes.shape)

        self.assertTrue(np.isfinite(k_baes).all())
        self.assertTrue(np.isfinite(k_const).all())
        np.testing.assert_allclose(k_baes, k_const, rtol=7e-3, atol=1e-6)

    def test_baes_kernel_reduces_to_constant_anisotropy_wide_beta_range(self):
        """BAES(beta_0=beta_inf) reproduces constant anisotropy for wide beta values."""
        beta_values = np.array([-10.0, -5.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9, 1.0])
        u = jnp.asarray(np.geomspace(1.0 + 1e-4, 120.0, 160), dtype=jnp.float32)[None, :]
        R = jnp.asarray([80.0, 250.0, 900.0], dtype=jnp.float32)[:, None]

        baes = BaesAnisotropyModel()
        const = ConstantAnisotropyModel()
        for beta_const in beta_values:
            params_baes = {
                "beta_0": float(beta_const),
                "beta_inf": float(beta_const),
                "r_a": 350.0,
                "eta": 2.0,
            }
            params_const = {"beta_ani": float(beta_const)}

            k_baes = np.asarray(baes.kernel(u, R, params=params_baes, n_kernel=320))
            k_const = np.asarray(const.kernel(u, R, params=params_const))
            k_const = np.broadcast_to(k_const, k_baes.shape)

            self.assertTrue(np.isfinite(k_baes).all())
            self.assertTrue(np.isfinite(k_const).all())
            np.testing.assert_allclose(
                k_baes,
                k_const,
                rtol=1.5e-2,
                atol=3e-6,
                err_msg=f"failed at beta={beta_const}",
            )

    def test_baes_kernel_reduces_to_constant_anisotropy_large_u(self):
        """Large-u regime should remain finite and consistent (regression for z->1 rounding)."""
        assert_baes_constant_large_u_consistency(beta_values=(-10.0, -5.0, -2.0, -1.0, 0.0, 0.5))

    def test_baes_kernel_constant_limit_does_not_call_constant_model(self):
        """BAES kernel should remain self-contained even when beta_0=beta_inf."""
        baes = BaesAnisotropyModel()
        u = jnp.asarray(np.geomspace(1.0 + 1e-4, 1e4, 128), dtype=jnp.float32)[None, :]
        R = jnp.asarray([1.0], dtype=jnp.float32)[:, None]
        params_baes = {
            "beta_0": -5.0,
            "beta_inf": -5.0,
            "r_a": 1.0,
            "eta": 1.0,
        }

        with patch("jeanspy.model_numpyro.ConstantAnisotropyModel.kernel", side_effect=RuntimeError("should not be called")):
            k_baes = np.asarray(baes.kernel(u, R, params=params_baes, n_kernel=512))

        self.assertTrue(np.isfinite(k_baes).all())

    def test_baes_kernel_n_kernel_sweep_accuracy_large_u_negative_beta(self):
        """Accuracy remains good across practical n_kernel values (speed/accuracy guard)."""
        baes = BaesAnisotropyModel()
        const = ConstantAnisotropyModel()

        u = jnp.asarray(np.geomspace(1.0 + 1e-4, 1e5, 260), dtype=jnp.float32)[None, :]
        R = jnp.asarray([1.0], dtype=jnp.float32)[:, None]
        params_baes = {
            "beta_0": -10.0,
            "beta_inf": -10.0,
            "r_a": 1.0,
            "eta": 1.0,
        }
        params_const = {"beta_ani": -10.0}

        k_ref = np.asarray(const.kernel(u, R, params=params_const)).reshape(-1)
        n_kernel_values = (64, 96, 128, 192, 256)
        max_rels = []
        for n_kernel in n_kernel_values:
            k_baes = np.asarray(baes.kernel(u, R, params=params_baes, n_kernel=n_kernel)).reshape(-1)
            self.assertTrue(np.isfinite(k_baes).all(), msg=f"non-finite at n_kernel={n_kernel}")
            rel = np.abs(k_baes - k_ref) / (np.abs(k_ref) + 1e-30)
            max_rels.append(float(np.max(rel)))

        # Even at low node count, large-u negative-beta case should stay accurate.
        self.assertLess(max(max_rels), 2e-4)

    def test_baes_kernel_matches_om_special_case(self):
        """BAES(beta_0=0,beta_inf=1,eta=2) matches the Osipkov-Merritt analytic kernel."""
        r_a = 500.0
        params_baes = {
            "beta_0": 0.0,
            "beta_inf": 1.0,
            "r_a": r_a,
            "eta": 2.0,
        }

        u = jnp.asarray(np.geomspace(1.0 + 1e-4, 80.0, 160), dtype=jnp.float32)[None, :]
        R = jnp.asarray([80.0, 200.0, 600.0], dtype=jnp.float32)[:, None]
        k_baes = np.asarray(BaesAnisotropyModel().kernel(u, R, params=params_baes, n_kernel=320))

        u_np = np.asarray(u)
        R_np = np.asarray(R)
        u2 = u_np**2
        u_a = r_a / R_np
        u2_a = u_a**2
        k_om = (
            (u2 + u2_a) * (u2_a + 0.5) / (u_np * (u2_a + 1.0) ** 1.5)
            * np.arctan(np.sqrt((u2 - 1.0) / (u2_a + 1.0)))
            - np.sqrt(1.0 - 1.0 / u2) / (2.0 * (u2_a + 1.0))
        )

        self.assertTrue(np.isfinite(k_baes).all())
        self.assertTrue(np.isfinite(k_om).all())
        np.testing.assert_allclose(k_baes, k_om, rtol=8e-3, atol=2e-6)

    def test_baes_om_special_case_matches_osipkov_merritt_model(self):
        """BAES OM-special-case kernel agrees with OsipkovMerrittModel implementation."""
        r_a = 450.0
        params_baes = {
            "beta_0": 0.0,
            "beta_inf": 1.0,
            "r_a": r_a,
            "eta": 2.0,
        }
        params_om = {"r_a": r_a}

        u = jnp.asarray(np.geomspace(1.0 + 1e-4, 120.0, 200), dtype=jnp.float32)[None, :]
        R = jnp.asarray([60.0, 180.0, 500.0, 1200.0], dtype=jnp.float32)[:, None]

        k_baes = np.asarray(BaesAnisotropyModel().kernel(u, R, params=params_baes, n_kernel=320))
        k_om_model = np.asarray(OsipkovMerrittModel().kernel(u, R, params=params_om))

        self.assertTrue(np.isfinite(k_baes).all())
        self.assertTrue(np.isfinite(k_om_model).all())
        np.testing.assert_allclose(k_baes, k_om_model, rtol=8e-3, atol=2e-6)

    def test_sigmalos2_matches_modelpy_reference(self):
        """Compare sigma_los^2 against the existing (SciPy/dequad) implementation.

        This guards against silent NaN/inf -> 0 sanitization regressions.
        """
        from jeanspy.model import (
            ConstantAnisotropyModel as ConstantAnisotropyModelRef,
            DSphModel as DSphModelRef,
            NFWModel as NFWModelRef,
            PlummerModel as PlummerModelRef,
            ZhaoModel as ZhaoModelRef,
        )

        cases = [
            {
                "name": "NFW",
                "dm_jax": NFWModel(),
                "dm_ref_ctor": lambda p: NFWModelRef(
                    rs_pc=p["rs_pc"],
                    rhos_Msunpc3=p["rhos_Msunpc3"],
                    r_t_pc=p["r_t_pc"],
                ),
                "params": {
                    "re_pc": 200.0,
                    "rs_pc": 1200.0,
                    "rhos_Msunpc3": 1e-2,
                    "r_t_pc": 8000.0,
                    "beta_ani": 0.2,
                    "vmem_kms": 0.0,
                },
                "rtol_max": 0.05,
            },
            {
                "name": "Zhao",
                "dm_jax": ZhaoModel(),
                "dm_ref_ctor": lambda p: ZhaoModelRef(
                    rs_pc=p["rs_pc"],
                    rhos_Msunpc3=p["rhos_Msunpc3"],
                    a=p["a"],
                    b=p["b"],
                    g=p["g"],
                    r_t_pc=p["r_t_pc"],
                ),
                "params": {
                    "re_pc": 220.0,
                    "rs_pc": 1000.0,
                    "rhos_Msunpc3": 8e-3,
                    "a": 1.1,
                    "b": 4.2,
                    "g": 0.7,
                    "r_t_pc": 9000.0,
                    "beta_ani": 0.2,
                    "vmem_kms": 0.0,
                },
                "rtol_max": 0.08,
            },
        ]

        R = np.array([5.0, 20.0, 100.0, 800.0], dtype=float)
        for case in cases:
            with self.subTest(dm_model=case["name"]):
                true = case["params"]
                dsph_jax = DSphModel(
                    submodels={
                        "StellarModel": PlummerModel(),
                        "DMModel": case["dm_jax"],
                        "AnisotropyModel": ConstantAnisotropyModel(),
                    }
                )
                dsph_ref = DSphModelRef(
                    submodels={
                        "StellarModel": PlummerModelRef(re_pc=true["re_pc"]),
                        "DMModel": case["dm_ref_ctor"](true),
                        "AnisotropyModel": ConstantAnisotropyModelRef(beta_ani=true["beta_ani"]),
                    },
                    vmem_kms=0.0,
                )

                s2_jax = np.array(dsph_jax.sigmalos2(jnp.asarray(R), params=true, n_u=512, u_max=3000.0))
                s2_ref = np.array(dsph_ref.sigmalos2_dequad(R, n=1024, n_kernel=128))

                self.assertTrue(np.isfinite(s2_jax).all())
                self.assertTrue(np.isfinite(s2_ref).all())
                self.assertTrue((s2_jax > 0).all())
                self.assertTrue((s2_ref > 0).all())

                rel = np.abs(s2_jax - s2_ref) / np.maximum(1e-12, np.abs(s2_ref))
                self.assertLess(float(np.max(rel)), case["rtol_max"])

    def test_sigmalos2_nfw_equals_zhao_nfw_limit(self):
        """Zhao(a=1,b=3,g=1) should reproduce NFW in sigmalos2."""
        params_base = {
            "re_pc": 200.0,
            "rs_pc": 1200.0,
            "rhos_Msunpc3": 1e-2,
            "r_t_pc": 8000.0,
            "beta_ani": 0.2,
            "vmem_kms": 0.0,
        }
        params_zhao = {**params_base, "a": 1.0, "b": 3.0, "g": 1.0}

        dsph_nfw = DSphModel(
            submodels={
                "StellarModel": PlummerModel(),
                "DMModel": NFWModel(),
                "AnisotropyModel": ConstantAnisotropyModel(),
            }
        )
        dsph_zhao = DSphModel(
            submodels={
                "StellarModel": PlummerModel(),
                "DMModel": ZhaoModel(),
                "AnisotropyModel": ConstantAnisotropyModel(),
            }
        )

        R = jnp.asarray(np.geomspace(1.0, 1e3, 96), dtype=jnp.float32)
        s2_nfw = np.asarray(dsph_nfw.sigmalos2(R, params=params_base, n_u=192, u_max=1500.0), dtype=np.float64)
        s2_zhao = np.asarray(dsph_zhao.sigmalos2(R, params=params_zhao, n_u=192, u_max=1500.0), dtype=np.float64)

        self.assertTrue(np.isfinite(s2_nfw).all())
        self.assertTrue(np.isfinite(s2_zhao).all())
        self.assertTrue((s2_nfw > 0).all())
        self.assertTrue((s2_zhao > 0).all())

        np.testing.assert_allclose(s2_zhao, s2_nfw, rtol=8e-3, atol=1e-8)

    def test_sigmalos2_nfw_equals_zhao_nfw_limit_sampled_R(self):
        """NFW and Zhao(a=1,b=3,g=1) agree on sampled projected radii (snippet parity)."""
        key = jax.random.PRNGKey(123)
        key, subkey = jax.random.split(key)

        params_nfw = {
            "re_pc": 200.0,
            "rs_pc": 1200.0,
            "rhos_Msunpc3": 1e-2,
            "r_t_pc": 8000.0,
            "beta_ani": 0.2,
            "vmem_kms": 0.0,
        }
        params_zhao = {**params_nfw, "a": 1.0, "b": 3.0, "g": 1.0}

        stellar = PlummerModel()
        nR = 1000
        R_pc = stellar.sample_R(subkey, nR, re_pc=params_nfw["re_pc"])

        dsph_nfw = DSphModel(
            submodels={
                "StellarModel": stellar,
                "DMModel": NFWModel(),
                "AnisotropyModel": ConstantAnisotropyModel(),
            }
        )
        dsph_zhao = DSphModel(
            submodels={
                "StellarModel": stellar,
                "DMModel": ZhaoModel(),
                "AnisotropyModel": ConstantAnisotropyModel(),
            }
        )

        @jax.jit
        def _sig2_nfw(R, p):
            return dsph_nfw.sigmalos2(R, params=p, n_u=192, u_max=1500.0)

        @jax.jit
        def _sig2_zhao(R, p):
            return dsph_zhao.sigmalos2(R, params=p, n_u=192, u_max=1500.0)

        s2_nfw = np.asarray(_sig2_nfw(jnp.asarray(R_pc), params_nfw), dtype=np.float64)
        s2_zhao = np.asarray(_sig2_zhao(jnp.asarray(R_pc), params_zhao), dtype=np.float64)

        self.assertTrue(np.isfinite(s2_nfw).all())
        self.assertTrue(np.isfinite(s2_zhao).all())
        self.assertTrue((s2_nfw > 0).all())
        self.assertTrue((s2_zhao > 0).all())

        np.testing.assert_allclose(s2_zhao, s2_nfw, rtol=1e-2, atol=1e-8)

    def test_sigmalos2_abel_matches_kernel_for_supported_anisotropy_models(self):
        R = jnp.asarray(np.geomspace(5.0, 900.0, 32), dtype=jnp.float32)

        for case in self._make_sigmalos2_backend_cases():
            with self.subTest(anisotropy=case["name"]):
                dsph = DSphModel(
                    submodels={
                        "StellarModel": PlummerModel(),
                        "DMModel": NFWModel(),
                        "AnisotropyModel": case["anisotropy"],
                    }
                )

                s2_kernel = np.asarray(
                    dsph.sigmalos2(
                        R,
                        params=case["params"],
                        backend="kernel",
                        n_u=224,
                        u_max=1600.0,
                    ),
                    dtype=np.float64,
                )
                s2_abel = np.asarray(
                    dsph.sigmalos2(
                        R,
                        params=case["params"],
                        backend="abel",
                        n_r=896,
                        u_max=1600.0,
                        r_min_factor=0.35,
                    ),
                    dtype=np.float64,
                )
                s2_auto = np.asarray(
                    dsph.sigmalos2(
                        R,
                        params=case["params"],
                        backend="auto",
                        n_u=224,
                        n_r=896,
                        u_max=1600.0,
                        r_min_factor=0.35,
                    ),
                    dtype=np.float64,
                )

                self.assertTrue(np.isfinite(s2_kernel).all(), msg=f"kernel backend non-finite for {case['name']}")
                self.assertTrue(np.isfinite(s2_abel).all(), msg=f"abel backend non-finite for {case['name']}")
                self.assertTrue(np.isfinite(s2_auto).all(), msg=f"auto backend non-finite for {case['name']}")
                self.assertTrue((s2_kernel > 0).all(), msg=f"kernel backend non-positive for {case['name']}")
                self.assertTrue((s2_abel > 0).all(), msg=f"abel backend non-positive for {case['name']}")

                # determine which explicit backend the auto choice corresponds to
                if np.allclose(s2_auto, s2_abel, rtol=1e-7, atol=0.0):
                    chosen = "abel"
                elif np.allclose(s2_auto, s2_kernel, rtol=1e-7, atol=0.0):
                    chosen = "kernel"
                else:
                    chosen = "<mismatch>"
                logger.info("auto backend selected %s", chosen)
                self.assertIn(chosen, {"abel", "kernel"})

                rel = np.abs(s2_abel - s2_kernel) / np.maximum(1e-10, np.abs(s2_kernel))
                logger.info(
                    "sigmalos2 backend comparison %s: max_rel=%.4e mean_rel=%.4e",
                    case["name"],
                    float(np.max(rel)),
                    float(np.mean(rel)),
                )
                self.assertLess(float(np.max(rel)), case["rtol_max"])

    def test_sigmalos2_backend_benchmark_reports_accuracy_and_speed(self):
        R = jnp.asarray(np.geomspace(5.0, 900.0, 48), dtype=jnp.float32)

        for case in self._make_sigmalos2_backend_cases():
            with self.subTest(anisotropy=case["name"]):
                dsph = DSphModel(
                    submodels={
                        "StellarModel": PlummerModel(),
                        "DMModel": NFWModel(),
                        "AnisotropyModel": case["anisotropy"],
                    }
                )

                kernel_fn = jax.jit(
                    lambda radii: dsph.sigmalos2(
                        radii,
                        params=case["params"],
                        backend="kernel",
                        n_u=192,
                        u_max=1400.0,
                    )
                )
                abel_fn = jax.jit(
                    lambda radii: dsph.sigmalos2(
                        radii,
                        params=case["params"],
                        backend="abel",
                        n_r=640,
                        u_max=1400.0,
                        r_min_factor=0.35,
                    )
                )
                auto_fn = jax.jit(
                    lambda radii: dsph.sigmalos2(
                        radii,
                        params=case["params"],
                        backend="auto",
                        n_u=192,
                        n_r=640,
                        u_max=1400.0,
                        r_min_factor=0.35,
                    )
                )

                kernel_ref = kernel_fn(R)
                abel_ref = abel_fn(R)
                auto_ref = auto_fn(R)
                jax.block_until_ready(kernel_ref)
                jax.block_until_ready(abel_ref)
                jax.block_until_ready(auto_ref)

                kernel_time = self._measure_jitted_runtime(kernel_fn, R)
                abel_time = self._measure_jitted_runtime(abel_fn, R)
                auto_time = self._measure_jitted_runtime(auto_fn, R)

                s2_kernel = np.asarray(kernel_ref, dtype=np.float64)
                s2_abel = np.asarray(abel_ref, dtype=np.float64)
                rel = np.abs(s2_abel - s2_kernel) / np.maximum(1e-10, np.abs(s2_kernel))
                speedup = kernel_time / abel_time if abel_time > 0.0 else np.inf

                # determine which backend auto_fn actually selected
                auto_res = np.asarray(auto_ref, dtype=np.float64)
                if np.allclose(auto_res, s2_abel, rtol=1e-7, atol=0.0):
                    auto_chosen = "abel"
                else:
                    auto_chosen = "kernel"

                logger.info(
                    "sigmalos2 benchmark %s: kernel=%.6fs abel=%.6fs auto=%.6fs (%s) speedup=%.2fx max_rel=%.4e mean_rel=%.4e",
                    case["name"],
                    kernel_time,
                    abel_time,
                    auto_time,
                    auto_chosen,
                    speedup,
                    float(np.max(rel)),
                    float(np.mean(rel)),
                )

                self.assertGreater(kernel_time, 0.0)
                self.assertGreater(abel_time, 0.0)
                self.assertTrue(np.isfinite(speedup))
                self.assertLess(float(np.max(rel)), max(case["rtol_max"], 9.0e-2))

    def test_zhao_betainc_enclosed_mass_consistent_with_numeric_default(self):
        """Keep betainc implementation and ensure it matches numeric enclosed mass."""
        zhao = ZhaoModel()
        params = {
            "rs_pc": 900.0,
            "rhos_Msunpc3": 8e-3,
            "a": 1.2,
            "b": 4.2,
            "g": 0.6,
            "r_t_pc": 8000.0,
        }
        r = jnp.asarray(np.geomspace(1.0, 6000.0, 64), dtype=jnp.float32)

        m_num = np.asarray(zhao.enclosed_mass(r, params=params), dtype=np.float64)
        m_beta = np.asarray(zhao.enclosed_mass_betainc(r, params=params), dtype=np.float64)

        self.assertTrue(np.isfinite(m_num).all())
        self.assertTrue(np.isfinite(m_beta).all())
        np.testing.assert_allclose(m_beta, m_num, rtol=2e-2, atol=1e-8)

    def test_model_submodels_validation(self):
        class MyModel1(Model):
            required_models = {}

        class MyModel2(Model):
            required_models = {}

        class MyModel12(Model):
            required_models = {"m1": MyModel1, "m2": MyModel2}

        # missing required submodels
        with self.assertRaises(ValueError):
            MyModel12(submodels={"m1": MyModel1()})

        # extra unexpected submodels
        with self.assertRaises(ValueError):
            MyModel12(submodels={"m1": MyModel1(), "m2": MyModel2(), "m3": MyModel1()})

        # correct
        MyModel12(submodels={"m1": MyModel1(), "m2": MyModel2()})

    def _arviz_smoke(self, mcmc: MCMC, *, var_names: list[str]):
        """Convert NumPyro MCMC results to ArviZ and generate a simple trace plot."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            idata = az.from_numpyro(mcmc)
            summ = az.summary(idata, var_names=var_names)
            self.assertTrue(len(summ) > 0)
            axes = az.plot_trace(idata, var_names=var_names)
            # Ensure matplotlib objects are created and then close.
            self.assertIsNotNone(axes)
            plt.close("all")

    def test_plummer_re_inference_nuts_and_aies(self):
        _log_runtime()
        numpyro.set_host_device_count(16)

        key = jax.random.PRNGKey(0)
        true_re = 200.0
        n = 256

        pl = PlummerModel()
        key, subkey = jax.random.split(key)
        R_obs = np.array(pl.sample_R(subkey, n, re_pc=true_re))

        def model(R_pc):
            # AD-friendly parameterization for NUTS
            log_re = numpyro.sample("log_re", dist.Normal(jnp.log(200.0), 0.8))
            re_pc = jnp.exp(log_re)
            logp = PlummerModel().log_prob_R(jnp.asarray(R_pc), re_pc=re_pc).sum()
            numpyro.factor("lik", logp)
            numpyro.deterministic("re_pc", re_pc)

        # --- NUTS demo ---
        logger.info("Running NUTS (warmup=%s, samples=%s, chains=%s)", 200, 200, 1)
        nuts = NUTS(model)
        mcmc = MCMC(nuts, num_warmup=200, num_samples=200, num_chains=1, progress_bar=False)
        mcmc.run(key, R_pc=R_obs)
        samples = mcmc.get_samples()
        re_mean = float(jnp.mean(samples["re_pc"]))
        self.assertLess(abs(re_mean - true_re) / true_re, 0.2)
        self._arviz_smoke(mcmc, var_names=["re_pc"])

        # --- AIES demo (gradient-free) ---
        # AIES uses chains as ensemble walkers.
        aies = AIES(model)
        num_chains = 16
        logger.info(
            "Running AIES (warmup=%s, samples=%s, chains=%s, chain_method=vectorized)",
            150,
            150,
            num_chains,
        )
        mcmc2 = MCMC(
            aies,
            num_warmup=150,
            num_samples=150,
            num_chains=num_chains,
            chain_method="vectorized",
            progress_bar=False,
        )
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_chains)
        mcmc2.run(keys, R_pc=R_obs)
        samples2 = mcmc2.get_samples(group_by_chain=False)
        re_mean2 = float(jnp.mean(samples2["re_pc"]))
        self.assertLess(abs(re_mean2 - true_re) / true_re, 0.3)
        self._arviz_smoke(mcmc2, var_names=["re_pc"])

    def test_jeans_demo_aies_runs(self):
        _log_runtime()
        numpyro.set_host_device_count(8)

        key = jax.random.PRNGKey(1)
        # True parameters for synthetic data
        true = {
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
                "AnisotropyModel": ConstantAnisotropyModel()
            }
        )

        # Observational setup
        nR = 24
        R_pc = jnp.geomspace(5.0, 800.0, nR)
        err = 2.0  # km/s
        err2 = (err * jnp.ones_like(R_pc)) ** 2

        # Generate synthetic vlos
        s2 = dsph.sigmalos2(R_pc, params=true, n_u=192, u_max=1500.0)
        key, subkey = jax.random.split(key)
        vlos = true["vmem_kms"] + jnp.sqrt(s2 + err2) * jax.random.normal(subkey, shape=R_pc.shape)
        
        # Mock observed data (convert to JAX arrays)
        R_obs = jnp.array(R_pc)
        v_obs = jnp.array(vlos)
        e_obs = jnp.array(err * jnp.ones_like(R_obs))

        # Define the model for inference using numpyro
        def model(R_pc, vlos_kms, e_vlos_kms):
            # Broad but finite priors (keep the demo stable)
            log_re = numpyro.sample("log_re", dist.Normal(jnp.log(200.0), 0.8))
            log_rs = numpyro.sample("log_rs", dist.Normal(jnp.log(1200.0), 0.8))
            log_rhos = numpyro.sample("log_rhos", dist.Normal(jnp.log(1e-2), 1.0))
            log_r_t = numpyro.sample("log_r_t", dist.Normal(jnp.log(8000.0), 0.4))
            beta_ani = numpyro.sample("beta_ani", dist.Uniform(-0.5, 0.8))
            vmem_kms = numpyro.sample("vmem_kms", dist.Normal(0.0, 30.0))
            # Deterministic transformations to physical parameters for interpretability
            re_pc = jnp.exp(log_re)
            rs_pc = jnp.exp(log_rs)
            rhos_Msunpc3 = jnp.exp(log_rhos)
            r_t_pc = jnp.exp(log_r_t)
            # Pack parameters for sigmalos2
            params = {
                "re_pc": re_pc,
                "rs_pc": rs_pc,
                "rhos_Msunpc3": rhos_Msunpc3,
                "r_t_pc": r_t_pc,
                "beta_ani": beta_ani,
                "vmem_kms": vmem_kms,
            }
            # Compute model-predicted sigma_los^2 and incorporate observational errors
            s2 = dsph.sigmalos2(jnp.asarray(R_pc), params=params, n_u=160, u_max=1500.0)
            s2 = jnp.clip(s2, min=1e-12, max=1e12)
            scale = jnp.sqrt(s2 + jnp.asarray(e_vlos_kms) ** 2)
            numpyro.sample("vlos", dist.Normal(params["vmem_kms"], scale), obs=jnp.asarray(vlos_kms))
            # Report physical params (easier to interpret than log-params)
            numpyro.deterministic("re_pc", re_pc)
            numpyro.deterministic("rs_pc", rs_pc)
            numpyro.deterministic("rhos_Msunpc3", rhos_Msunpc3)
            numpyro.deterministic("r_t_pc", r_t_pc)

        # AIES demo: keep short (runtime-sensitive)
        aies = AIES(model)
        num_chains = 8
        logger.info(
            "Running Jeans AIES (warmup=%s, samples=%s, chains=%s, chain_method=vectorized)",
            60,
            60,
            num_chains,
        )
        mcmc = MCMC(
            aies,
            num_warmup=60,
            num_samples=60,
            num_chains=num_chains,
            chain_method="vectorized",
            progress_bar=False,
        )
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_chains)
        mcmc.run(keys, R_pc=R_obs, vlos_kms=v_obs, e_vlos_kms=e_obs)
        samples = mcmc.get_samples(group_by_chain=False)

        # Sanity checks: samples exist and are finite
        for name in ("re_pc", "rs_pc", "rhos_Msunpc3", "r_t_pc", "beta_ani", "vmem_kms"):
            self.assertIn(name, samples)
            self.assertTrue(np.isfinite(np.array(samples[name])).all())

        self._arviz_smoke(
            mcmc,
            var_names=["re_pc", "rs_pc", "rhos_Msunpc3", "r_t_pc", "beta_ani", "vmem_kms"],
        )

    def test_jeans_demo_nuts_runs_jax_backend(self):
        """Jeans demo should run with NUTS when hypergeometric backend is JAX."""
        _log_runtime()

        key = jax.random.PRNGKey(2)
        true = {
            "re_pc": 200.0,
            "rs_pc": 1200.0,
            "rhos_Msunpc3": 1e-2,
            "r_t_pc": 8000.0,
            "beta_ani": 0.2,
            "vmem_kms": 0.0,
        }

        with patch.object(model_numpyro_mod, "_HYP2F1_BACKEND", "jax"):
            dsph = DSphModel(
                submodels={
                    "StellarModel": PlummerModel(),
                    "DMModel": NFWModel(),
                    "AnisotropyModel": ConstantAnisotropyModel(),
                }
            )

            nR = 20
            R_pc = jnp.geomspace(5.0, 700.0, nR)
            err = 2.0
            err2 = (err * jnp.ones_like(R_pc)) ** 2

            s2_true = dsph.sigmalos2(R_pc, params=true, n_u=144, u_max=1200.0)
            key, subkey = jax.random.split(key)
            vlos = true["vmem_kms"] + jnp.sqrt(s2_true + err2) * jax.random.normal(subkey, shape=R_pc.shape)

            R_obs = jnp.array(R_pc)
            v_obs = jnp.array(vlos)
            e_obs = jnp.array(err * jnp.ones_like(R_obs))

            def model(R_pc, vlos_kms, e_vlos_kms):
                log_re = numpyro.sample("log_re", dist.Normal(jnp.log(200.0), 0.8))
                log_rs = numpyro.sample("log_rs", dist.Normal(jnp.log(1200.0), 0.8))
                log_rhos = numpyro.sample("log_rhos", dist.Normal(jnp.log(1e-2), 1.0))
                log_r_t = numpyro.sample("log_r_t", dist.Normal(jnp.log(8000.0), 0.5))
                beta_ani = numpyro.sample("beta_ani", dist.Uniform(-0.4, 0.8))
                vmem_kms = numpyro.sample("vmem_kms", dist.Normal(0.0, 30.0))

                re_pc = jnp.exp(log_re)
                rs_pc = jnp.exp(log_rs)
                rhos_Msunpc3 = jnp.exp(log_rhos)
                r_t_pc = jnp.exp(log_r_t)

                params = {
                    "re_pc": re_pc,
                    "rs_pc": rs_pc,
                    "rhos_Msunpc3": rhos_Msunpc3,
                    "r_t_pc": r_t_pc,
                    "beta_ani": beta_ani,
                    "vmem_kms": vmem_kms,
                }

                s2 = dsph.sigmalos2(jnp.asarray(R_pc), params=params, n_u=128, u_max=1200.0)
                s2 = jnp.clip(s2, min=1e-12, max=1e12)
                scale = jnp.sqrt(s2 + jnp.asarray(e_vlos_kms) ** 2)

                numpyro.sample("vlos", dist.Normal(vmem_kms, scale), obs=jnp.asarray(vlos_kms))
                numpyro.deterministic("re_pc", re_pc)
                numpyro.deterministic("rs_pc", rs_pc)
                numpyro.deterministic("rhos_Msunpc3", rhos_Msunpc3)
                numpyro.deterministic("r_t_pc", r_t_pc)

            nuts = NUTS(model)
            mcmc = MCMC(nuts, num_warmup=80, num_samples=80, num_chains=1, progress_bar=False)
            key, subkey = jax.random.split(key)
            mcmc.run(subkey, R_pc=R_obs, vlos_kms=v_obs, e_vlos_kms=e_obs)
            samples = mcmc.get_samples()

            for name in ("re_pc", "rs_pc", "rhos_Msunpc3", "r_t_pc", "beta_ani", "vmem_kms"):
                self.assertIn(name, samples)
                self.assertTrue(np.isfinite(np.array(samples[name])).all())

            re_mean = float(np.mean(np.array(samples["re_pc"])))
            self.assertLess(abs(re_mean - true["re_pc"]) / true["re_pc"], 0.6)

            self._arviz_smoke(
                mcmc,
                var_names=["re_pc", "rs_pc", "rhos_Msunpc3", "r_t_pc", "beta_ani", "vmem_kms"],
            )


if __name__ == "__main__":
    unittest.main()
