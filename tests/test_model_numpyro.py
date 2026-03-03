import unittest
import warnings

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

from scipy.special import hyp2f1 as scipy_hyp2f1

from jeanspy.model_numpyro import ConstantAnisotropyModel, DSphModel, Model, NFWModel, PlummerModel


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

    def test_sigmalos2_matches_modelpy_reference(self):
        """Compare sigma_los^2 against the existing (SciPy/dequad) implementation.

        This guards against silent NaN/inf -> 0 sanitization regressions.
        """
        from jeanspy.model import (
            ConstantAnisotropyModel as ConstantAnisotropyModelRef,
            DSphModel as DSphModelRef,
            NFWModel as NFWModelRef,
            PlummerModel as PlummerModelRef,
        )

        true = {
            "re_pc": 200.0,
            "rs_pc": 1200.0,
            "rhos_Msunpc3": 1e-2,
            "r_t_pc": 8000.0,
            "beta_ani": 0.2,
            "vmem_kms": 0.0,
        }

        # JAX version
        dsph_jax = DSphModel(
            submodels={
                "StellarModel": PlummerModel(),
                "DMModel": NFWModel(),
                "AnisotropyModel": ConstantAnisotropyModel(),
            }
        )

        # Reference version (SciPy/dequad)
        dsph_ref = DSphModelRef(
            submodels={
                "StellarModel": PlummerModelRef(re_pc=true["re_pc"]),
                "DMModel": NFWModelRef(
                    rs_pc=true["rs_pc"],
                    rhos_Msunpc3=true["rhos_Msunpc3"],
                    r_t_pc=true["r_t_pc"],
                ),
                "AnisotropyModel": ConstantAnisotropyModelRef(beta_ani=true["beta_ani"]),
            },
            vmem_kms=0.0,
        )

        R = np.array([5.0, 20.0, 100.0, 800.0], dtype=float)
        s2_jax = np.array(dsph_jax.sigmalos2(jnp.asarray(R), params=true, n_u=512, u_max=3000.0))
        s2_ref = np.array(dsph_ref.sigmalos2_dequad(R, n=1024, n_kernel=128))

        self.assertTrue(np.isfinite(s2_jax).all())
        self.assertTrue(np.isfinite(s2_ref).all())
        self.assertTrue((s2_jax > 0).all())
        self.assertTrue((s2_ref > 0).all())

        rel = np.abs(s2_jax - s2_ref) / np.maximum(1e-12, np.abs(s2_ref))
        self.assertLess(float(np.max(rel)), 0.05)

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

        stellar = PlummerModel()
        dm = NFWModel()
        ani = ConstantAnisotropyModel()
        dsph = DSphModel(submodels={"StellarModel": stellar, "DMModel": dm, "AnisotropyModel": ani})

        # Observational setup
        nR = 24
        R_pc = jnp.geomspace(5.0, 800.0, nR)
        err = 2.0  # km/s
        err2 = (err * jnp.ones_like(R_pc)) ** 2

        # Generate synthetic vlos
        s2 = dsph.sigmalos2(R_pc, params=true, n_u=192, u_max=1500.0)
        key, subkey = jax.random.split(key)
        vlos = true["vmem_kms"] + jnp.sqrt(s2 + err2) * jax.random.normal(subkey, shape=R_pc.shape)

        R_obs = np.array(R_pc)
        v_obs = np.array(vlos)
        e_obs = np.array(err * np.ones_like(R_obs))

        def model(R_pc, vlos_kms, e_vlos_kms):
            # Broad but finite priors (keep the demo stable)
            log_re = numpyro.sample("log_re", dist.Normal(jnp.log(200.0), 0.8))
            log_rs = numpyro.sample("log_rs", dist.Normal(jnp.log(1200.0), 0.8))
            log_rhos = numpyro.sample("log_rhos", dist.Normal(jnp.log(1e-2), 1.0))
            log_r_t = numpyro.sample("log_r_t", dist.Normal(jnp.log(8000.0), 0.4))
            beta_ani = numpyro.sample("beta_ani", dist.Uniform(-0.5, 0.8))
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


if __name__ == "__main__":
    unittest.main()
