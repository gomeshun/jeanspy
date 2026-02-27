from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

# IMPORTANT: these env vars must be set before importing JAX.
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# A small default safety fraction; users can override via env.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jeanspy.hyp2f1_jax import hyp2f1_1b_3half

# Optional logger (used by callers/tests; kept lightweight)
import logging

logger = logging.getLogger(__name__)


_HYP2F1_BACKEND = os.environ.get("JEANSPY_HYP2F1_BACKEND", "scipy").strip().lower()
_HYP2F1_JAX_METHOD = os.environ.get("JEANSPY_HYP2F1_JAX_METHOD", "auto").strip().lower()
_HYP2F1_JAX_N_TERMS = int(os.environ.get("JEANSPY_HYP2F1_JAX_N_TERMS", "192"))
_HYP2F1_JAX_N_QUAD = int(os.environ.get("JEANSPY_HYP2F1_JAX_N_QUAD", "128"))
_HYP2F1_JAX_QUAD_RULE = os.environ.get("JEANSPY_HYP2F1_JAX_QUAD_RULE", "tanh_sinh").strip().lower()

if _HYP2F1_BACKEND not in {"scipy", "jax"}:
    raise ValueError(
        "JEANSPY_HYP2F1_BACKEND must be one of {'scipy','jax'} "
        f"but got {_HYP2F1_BACKEND!r}"
    )

if _HYP2F1_JAX_QUAD_RULE not in {"tanh_sinh", "gauss_kronrod"}:
    raise ValueError(
        "JEANSPY_HYP2F1_JAX_QUAD_RULE must be one of {'tanh_sinh','gauss_kronrod'} "
        f"but got {_HYP2F1_JAX_QUAD_RULE!r}"
    )


def _hyp2f1_1_b_3half_scipy(z: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Compute hyp2f1(1, b; 3/2; z) using SciPy via jax.pure_callback.

        Rationale:
        - jax.scipy.special.hyp2f1 can be inaccurate/unstable for some parameter
            regimes relevant to Jeans kernels.
        - Our default MCMC for these models is gradient-free (AIES), so a host
            callback is acceptable and more robust.
        """
        import numpy as _np
        from scipy.special import hyp2f1 as _scipy_hyp2f1

        z = jnp.asarray(z)
        b = jnp.asarray(b)

        out_spec = jax.ShapeDtypeStruct(z.shape, z.dtype)

        def _cb(z_np, b_np):
                # b is expected to be scalar in our current usage.
                b_val = float(_np.asarray(b_np).reshape(()))
                z_arr = _np.asarray(z_np)
                return _scipy_hyp2f1(1.0, b_val, 1.5, z_arr).astype(z_arr.dtype, copy=False)

        return jax.pure_callback(_cb, out_spec, z, b, vmap_method="sequential")


def _hyp2f1_1_b_3half_jax(z: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Compute hyp2f1(1, b; 3/2; z) using jeanspy JAX implementation."""
        z = jnp.asarray(z)
        b = jnp.asarray(b)
        return hyp2f1_1b_3half(
            b,
            z,
            method=_HYP2F1_JAX_METHOD,
            n_terms=_HYP2F1_JAX_N_TERMS,
            n_quad=_HYP2F1_JAX_N_QUAD,
            quad_rule=_HYP2F1_JAX_QUAD_RULE,
        )


def _hyp2f1_1_b_3half(z: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        if _HYP2F1_BACKEND == "jax":
            return _hyp2f1_1_b_3half_jax(z, b)
        return _hyp2f1_1_b_3half_scipy(z, b)


# Physical constants (floats are fine in JAX computations)
GMsun_m3s2: float = 1.32712440018e20
PARSEC_M: float = 3.085677581491367e16


def _trapz(y: jnp.ndarray, x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """JAX-friendly trapezoidal integration."""
    dx = jnp.diff(x, axis=axis)
    y0 = jnp.take(y, indices=jnp.arange(y.shape[axis] - 1), axis=axis)
    y1 = jnp.take(y, indices=jnp.arange(1, y.shape[axis]), axis=axis)
    return jnp.sum((y0 + y1) * 0.5 * dx, axis=axis)


class Model:
    """Minimal model container designed to work well with NumPyro.

    Design goals:
    - Keep parameter grouping + submodel composition (like the existing model.py)
    - Avoid pandas/scipy and keep computations JAX-friendly
    - Keep side effects minimal: prefer passing params explicitly in numpyro models

    Notes:
    - This base class intentionally implements only what is needed for MCMC demos/tests.
    """

    # Kept as metadata only; NumPyro models typically pass params explicitly.
    required_param_names: tuple[str, ...] = ()
    required_models: Mapping[str, type["Model"]] = {}

    def __init__(self, submodels: Optional[Mapping[str, "Model"]] = None):
        if submodels is None:
            submodels = {}

        # Validate required submodels
        expected = set(getattr(self, "required_models", {}).keys())
        provided = set(submodels.keys())
        if expected != provided:
            raise ValueError(
                f"{self.__class__.__name__} requires submodels {sorted(expected)} "
                f"but got {sorted(provided)}"
            )
        self.submodels: Dict[str, Model] = dict(submodels)

    def __getitem__(self, key: str) -> "Model":
        return self.submodels[key]


class StellarModel(Model):
    required_models: Mapping[str, type[Model]] = {}

    def surface_density_2d(self, R_pc: jnp.ndarray, *, re_pc: Any) -> jnp.ndarray:
        raise NotImplementedError

    def volume_density_3d(self, r_pc: jnp.ndarray, *, re_pc: Any) -> jnp.ndarray:
        raise NotImplementedError


class PlummerModel(StellarModel):
    r"""Plummer surface/volume density normalized so that \int 2πR Σ(R) dR = 1."""

    required_param_names = ("re_pc",)

    def surface_density_2d(self, R_pc: jnp.ndarray, *, re_pc: Any) -> jnp.ndarray:
        re = jnp.asarray(re_pc)
        x2 = (R_pc / re) ** 2
        return 1.0 / (jnp.pi * re**2) * (1.0 + x2) ** (-2.0)

    def volume_density_3d(self, r_pc: jnp.ndarray, *, re_pc: Any) -> jnp.ndarray:
        re = jnp.asarray(re_pc)
        x2 = (r_pc / re) ** 2
        return (3.0 / (4.0 * jnp.pi * re**3)) * (1.0 + x2) ** (-2.5)

    def log_prob_R(self, R_pc: jnp.ndarray, *, re_pc: Any) -> jnp.ndarray:
        """Log-pdf of observed projected radius R (i.e. p(R) dR).

        p(R) = 2πR Σ(R) = 2R/re^2 * (1 + (R/re)^2)^(-2)
        """
        re = jnp.asarray(re_pc)
        R = jnp.asarray(R_pc)
        x2 = (R / re) ** 2
        logp = jnp.log(2.0) + jnp.log(R) - 2.0 * jnp.log(re) - 2.0 * jnp.log1p(x2)
        return logp

    def sample_R(self, key: jax.Array, n: int, *, re_pc: Any) -> jnp.ndarray:
        """Sample projected radii using the analytic inverse CDF.

        CDF(R) = R^2 / (R^2 + re^2)  ->  R = re * sqrt(u/(1-u)).
        """
        u = jax.random.uniform(key, shape=(n,), minval=0.0, maxval=1.0)
        u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
        re = jnp.asarray(re_pc)
        return re * jnp.sqrt(u / (1.0 - u))


class DMModel(Model):
    required_models: Mapping[str, type[Model]] = {}

    def enclosed_mass(self, r_pc: jnp.ndarray, *, rs_pc: Any, rhos_Msunpc3: Any, r_t_pc: Any) -> jnp.ndarray:
        raise NotImplementedError


class NFWModel(DMModel):
    required_param_names = ("rs_pc", "rhos_Msunpc3", "r_t_pc")

    def enclosed_mass(self, r_pc: jnp.ndarray, *, rs_pc: Any, rhos_Msunpc3: Any, r_t_pc: Any) -> jnp.ndarray:
        rs = jnp.asarray(rs_pc)
        rhos = jnp.asarray(rhos_Msunpc3)
        r_t = jnp.asarray(r_t_pc)

        r = jnp.minimum(jnp.asarray(r_pc), r_t)
        x = r / rs
        # M(r) = 4π ρs rs^3 [ ln(1+x) - x/(1+x) ]
        coeff = 4.0 * jnp.pi * rhos * rs**3
        return coeff * (jnp.log1p(x) - x / (1.0 + x))


class AnisotropyModel(Model):
    required_models: Mapping[str, type[Model]] = {}

    def beta(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        raise NotImplementedError

    def kernel(self, u: jnp.ndarray, R_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        raise NotImplementedError


class ConstantAnisotropyModel(AnisotropyModel):
    required_param_names = ("beta_ani",)

    def beta(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        return jnp.asarray(params["beta_ani"])

    def kernel(self, u: jnp.ndarray, R_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        """LOSVD kernel K(u) for constant anisotropy.

        This is JAX-compatible via jax.scipy.special.hyp2f1.
        """
        beta_ani = jnp.asarray(params["beta_ani"])

        # The original expression (as in model.py) uses
        #   hyp2f1(1, 1.5-beta, 1.5, 1-u^2)
        # where (1-u^2) can be a large negative number for large u.
        # jax.scipy.special.hyp2f1 is numerically unstable in that regime.
        #
        # Use the exact hypergeometric transformation:
        #   2F1(a,b;c;z) = (1-z)^(-a) * 2F1(a, c-b; c; z/(z-1))
        # with a=1, b=1.5-beta, c=1.5, z=1-u^2.
        # Then (1-z)=u^2 and z/(z-1)=1-1/u^2 in (0,1) for u>1.
        u2 = u**2
        z_stable = 1.0 - 1.0 / u2
        hyp_stable = _hyp2f1_1_b_3half(z_stable, beta_ani)

        # sqrt term is well-defined for u>1. We avoid u=1 exactly in integration grids.
        pref = jnp.sqrt(1.0 - 1.0 / u2)
        return pref * ((1.5 - beta_ani) * hyp_stable - 0.5)


class DSphModel(Model):
    """Minimal Jeans model for sigma_los(R) using a fixed quadrature grid.

    This implementation is intentionally limited:
    - StellarModel: currently assumed to be PlummerModel-compatible interface
    - DMModel: currently assumed NFW-like enclosed mass interface
    - AnisotropyModel: constant anisotropy kernel (hyp2f1)

    It is sufficient for demonstrating MCMC with AIES/NUTS in tests.
    """

    required_param_names = ("vmem_kms",)
    required_models = {
        "StellarModel": StellarModel,
        "DMModel": DMModel,
        "AnisotropyModel": AnisotropyModel,
    }

    def sigmalos2(
        self,
        R_pc: jnp.ndarray,
        *,
        params: Mapping[str, Any],
        n_u: int = 256,
        u_max: float = 2e3,
        u_min_eps: float = 1e-6,
    ) -> jnp.ndarray:
        """Compute sigma_los^2(R) for an array of projected radii R_pc."""
        R = jnp.atleast_1d(jnp.asarray(R_pc))

        # Integration grid in log(u)
        t = jnp.linspace(jnp.log1p(u_min_eps), jnp.log(u_max), n_u)
        u = jnp.exp(t)  # (n_u,)

        # Broadcast shapes: (n_R, n_u)
        R2d = R[:, None]
        u2d = u[None, :]
        r = R2d * u2d

        stellar: StellarModel = self.submodels["StellarModel"]  # type: ignore[assignment]
        dm: DMModel = self.submodels["DMModel"]  # type: ignore[assignment]
        ani: AnisotropyModel = self.submodels["AnisotropyModel"]  # type: ignore[assignment]

        re_pc = params["re_pc"]
        rs_pc = params["rs_pc"]
        rhos = params["rhos_Msunpc3"]
        r_t = params["r_t_pc"]

        nu3 = stellar.volume_density_3d(r, re_pc=re_pc)
        sig2 = stellar.surface_density_2d(R2d, re_pc=re_pc)
        M = dm.enclosed_mass(r, rs_pc=rs_pc, rhos_Msunpc3=rhos, r_t_pc=r_t)

        K = ani.kernel(u2d, R2d, params=params)

        # Following the structure in the original model.py (unit-carrying constants kept)
        integrand_u = (
            2.0
            * (K / u2d)
            * (nu3 / sig2)
            * (GMsun_m3s2 * M / PARSEC_M)
            * 1e-6
        )

        # du = exp(t) dt  -> integrate over t
        integrand_t = integrand_u * u2d
        out = _trapz(integrand_t, t[None, :], axis=-1)
        # Numerical safety: sigma_los^2 should be >=0, but coarse quadrature / edge params
        # can produce tiny negatives or NaNs during MCMC initialization.
        out = jnp.nan_to_num(out, nan=0.0, neginf=0.0, posinf=1e12)
        return jnp.clip(out, min=0.0, max=1e12)


__all__ = [
    "Model",
    "PlummerModel",
    "NFWModel",
    "ConstantAnisotropyModel",
    "DSphModel",
]
