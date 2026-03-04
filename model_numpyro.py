from __future__ import annotations

from functools import lru_cache, partial
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
import numpy as np
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


def _hyp2f1_1_b_3half_from_u_scipy(u: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Compute hyp2f1(1, b; 3/2; 1-1/u^2) robustly for large u via SciPy callback.

        We compute z=1-1/u^2 in float64 on host to avoid float32 cancellation that
        rounds z to exactly 1 when u is very large.
        """
        import numpy as _np
        from scipy.special import hyp2f1 as _scipy_hyp2f1

        u = jnp.asarray(u)
        b = jnp.asarray(b)
        out_spec = jax.ShapeDtypeStruct(u.shape, u.dtype)

        def _cb(u_np, b_np):
            b_val = float(_np.asarray(b_np).reshape(()))
            u_arr = _np.asarray(u_np, dtype=_np.float64)
            u2 = u_arr * u_arr
            z_arr = 1.0 - 1.0 / u2
            z_arr = _np.clip(z_arr, 0.0, _np.nextafter(1.0, 0.0))
            return _scipy_hyp2f1(1.0, b_val, 1.5, z_arr).astype(_np.asarray(u_np).dtype, copy=False)

        return jax.pure_callback(_cb, out_spec, u, b, vmap_method="sequential")


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
        z = jnp.asarray(z)
        # Physical domain is z in [0, 1). For very large u in float32,
        # z=1-1/u^2 can round to exactly 1 and spuriously diverge.
        one = jnp.asarray(1.0, dtype=z.dtype)
        zero = jnp.asarray(0.0, dtype=z.dtype)
        z_upper = jnp.nextafter(one, zero)
        z = jnp.clip(z, zero, z_upper)

        if _HYP2F1_BACKEND == "jax":
            return _hyp2f1_1_b_3half_jax(z, b)
        return _hyp2f1_1_b_3half_scipy(z, b)


@partial(jax.jit, static_argnames=("n_terms_regular",))
def _hyp2f1_1_b_3half_asymptotic_from_u(
    u: jnp.ndarray,
    b: jnp.ndarray,
    *,
    n_terms_regular: int = 4,
    b_half_tol: float = 1e-6,
) -> jnp.ndarray:
        """Asymptotic 2F1(1,b;3/2;1-1/u^2) evaluated from u directly."""
        u_arr = jnp.asarray(u)
        b_arr = jnp.asarray(b)
        dtype = jnp.result_type(u_arr, b_arr)
        u_arr = u_arr.astype(dtype)
        b_arr = b_arr.astype(dtype)

        one = jnp.asarray(1.0, dtype=dtype)
        half = jnp.asarray(0.5, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        finfo = jnp.finfo(dtype)
        tiny = jnp.asarray(finfo.tiny, dtype=dtype)
        eps = jnp.asarray(finfo.eps, dtype=dtype)

        u_safe = jnp.maximum(u_arr, one + eps)
        delta = jnp.clip(one / (u_safe * u_safe), tiny, one - eps)
        sqrt_w = jnp.sqrt(one - delta)

        # b=1/2 exact branch with stable computation of (1-sqrt_w).
        one_minus_sqrt_w = delta / (one + sqrt_w)
        val_b_half = half * (jnp.log1p(sqrt_w) - jnp.log(one_minus_sqrt_w)) / sqrt_w
        is_b_half = jnp.abs(b_arr - half) <= jnp.asarray(b_half_tol, dtype=dtype)

        b_safe = jnp.where(is_b_half, b_arr + jnp.asarray(b_half_tol, dtype=dtype), b_arr)

        # Regular factor: 2F1(1,b;b+1/2;delta)
        term = jnp.ones_like(delta, dtype=dtype)
        regular_factor = term
        n_reg = max(int(n_terms_regular), 1)
        for i in range(n_reg - 1):
            i_f = jnp.asarray(i, dtype=dtype)
            term = term * (b_safe + i_f) / (b_safe + half + i_f) * delta
            regular_factor = regular_factor + term

        a_const = one / (one - two * b_safe)
        regular_term = a_const * regular_factor

        log_b_const = (
            jsp.gammaln(jnp.asarray(1.5, dtype=dtype))
            + jsp.gammaln(b_safe - half)
            - jsp.gammaln(b_safe)
        )
        b_const_sign = jsp.gammasgn(b_safe - half)
        singular_term = b_const_sign * jnp.exp(log_b_const + (half - b_safe) * jnp.log(delta)) / sqrt_w

        val_general = regular_term + singular_term
        return jnp.asarray(jnp.where(is_b_half, val_b_half, val_general), dtype=dtype)

@jax.jit
def _constant_kernel_jax_backend(u: jnp.ndarray, beta_ani: jnp.ndarray) -> jnp.ndarray:
        """Fast JAX-path constant-anisotropy kernel."""
        u_arr = jnp.asarray(u)
        dtype = jnp.result_type(u_arr, beta_ani)
        one = jnp.asarray(1.0, dtype=dtype)
        eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)

        beta = jnp.asarray(beta_ani, dtype=dtype)
        u_in = u_arr.astype(dtype)
        u_safe = jnp.maximum(u_in, one + eps)
        u2 = u_safe**2

        z_stable = 1.0 - 1.0 / u2
        hyp_main = _hyp2f1_1_b_3half_jax(z_stable, beta)
        hyp_asym = _hyp2f1_1_b_3half_asymptotic_from_u(u_safe, beta, n_terms_regular=4)

        use_asym = ((u_safe > jnp.asarray(20.0, dtype=dtype)) & (beta > 0.0) & (beta < 1.49)) | (~jnp.isfinite(hyp_main))
        hyp_stable = jnp.where(use_asym, hyp_asym, hyp_main)

        pref = jnp.sqrt(1.0 - 1.0 / u2)
        kernel_val = pref * ((1.5 - beta) * hyp_stable - 0.5)
        return jnp.where(u_in <= 1.0, jnp.zeros_like(kernel_val), kernel_val)


@jax.jit
def _osipkov_kernel_jax_backend(u: jnp.ndarray, r_pc: jnp.ndarray, r_a: jnp.ndarray) -> jnp.ndarray:
        """Fast JAX-path Osipkov-Merritt kernel."""
        u_arr, r_arr = jnp.broadcast_arrays(jnp.asarray(u), jnp.asarray(r_pc))
        dtype = jnp.result_type(u_arr, r_arr, r_a)
        one = jnp.asarray(1.0, dtype=dtype)
        eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)

        u_in = u_arr.astype(dtype)
        u_safe = jnp.maximum(u_in, one + eps)
        r_safe = r_arr.astype(dtype)
        r_a_safe = jnp.asarray(r_a, dtype=dtype)

        u_a = r_a_safe / r_safe
        u2_a = u_a**2
        u2 = u_safe**2
        sqrt_term = jnp.sqrt((u2 - 1.0) / (u2_a + 1.0))
        arctan_term = jnp.arctan(sqrt_term)
        sqrt_term2 = jnp.sqrt(1.0 - 1.0 / u2)
        kernel_val = (
            (u2 + u2_a) * (u2_a + 0.5) / (u_safe * (u2_a + 1.0) ** 1.5) * arctan_term
            - sqrt_term2 / (2.0 * (u2_a + 1.0))
        )
        return jnp.where(u_in <= 1.0, jnp.zeros_like(kernel_val), kernel_val)


@partial(jax.jit, static_argnames=("n_kernel",))
def _baes_kernel_jax_backend(
    u: jnp.ndarray,
    r_pc: jnp.ndarray,
    beta_0: jnp.ndarray,
    beta_inf: jnp.ndarray,
    r_a: jnp.ndarray,
    eta: jnp.ndarray,
    *,
    n_kernel: int,
) -> jnp.ndarray:
        """Fast JAX-path BAES kernel using Gauss-Legendre quadrature in arccosh(u)."""
        u_arr, r_arr = jnp.broadcast_arrays(jnp.asarray(u), jnp.asarray(r_pc))
        dtype = jnp.result_type(u_arr, r_arr, beta_0, beta_inf, r_a, eta)
        one = jnp.asarray(1.0, dtype=dtype)
        eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)

        u_in = u_arr.astype(dtype)
        u_safe = jnp.maximum(u_in, one + eps)
        r_safe = r_arr.astype(dtype)
        beta0 = jnp.asarray(beta_0, dtype=dtype)
        betainf = jnp.asarray(beta_inf, dtype=dtype)
        r_a_safe = jnp.asarray(r_a, dtype=dtype)
        eta_safe = jnp.asarray(eta, dtype=dtype)

        s_max = jnp.arccosh(u_safe)
        n_nodes = max(int(n_kernel), 8)
        t01, w01 = _gauss_legendre_01(n_nodes)
        t01 = jnp.asarray(t01, dtype=dtype)
        w01 = jnp.asarray(w01, dtype=dtype)

        s = s_max[..., None] * t01[None, ...]
        u_int = jnp.cosh(s)
        r_int = r_safe[..., None] * u_int

        x_int = (r_int / r_a_safe) ** eta_safe
        beta_int = (beta0 + betainf * x_int) / (1.0 + x_int)
        log_f_int = 2.0 * beta0 * jnp.log(r_int) + (2.0 * (betainf - beta0) / eta_safe) * jnp.log1p(x_int)

        r_s = r_safe * u_safe
        x_s = (r_s / r_a_safe) ** eta_safe
        log_f_s = 2.0 * beta0 * jnp.log(r_s) + (2.0 * (betainf - beta0) / eta_safe) * jnp.log1p(x_s)

        log_ratio = jnp.clip(log_f_s[..., None] - log_f_int, min=-80.0, max=80.0)
        integ_s = u_int * (1.0 - beta_int / (u_int**2)) * jnp.exp(log_ratio)
        weights = s_max[..., None] * w01[None, ...]
        inner = jnp.sum(weights * integ_s, axis=-1)

        k_val = inner / u_safe
        k_val = jnp.where(u_in <= 1.0, jnp.zeros_like(k_val), k_val)
        return jnp.nan_to_num(k_val, nan=0.0, neginf=0.0, posinf=1e12)


# Physical constants (floats are fine in JAX computations)
GMsun_m3s2: float = 1.32712440018e20
PARSEC_M: float = 3.085677581491367e16


@lru_cache(maxsize=16)
def _gauss_legendre_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights mapped to [0,1]."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    t = 0.5 * (x + 1.0)
    wt = 0.5 * w
    return t, wt


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

    def density_2d(self, R_pc: jnp.ndarray, *, re_pc: Any) -> jnp.ndarray:
        raise NotImplementedError

    def density_3d(self, r_pc: jnp.ndarray, *, re_pc: Any) -> jnp.ndarray:
        raise NotImplementedError


class PlummerModel(StellarModel):
    r"""Plummer surface/volume density normalized so that \int 2πR Σ(R) dR = 1."""

    required_param_names = ("re_pc",)

    def density_2d(self, R_pc: jnp.ndarray, *, re_pc: Any) -> jnp.ndarray:
        re = jnp.asarray(re_pc)
        x2 = (R_pc / re) ** 2
        return 1.0 / (jnp.pi * re**2) * (1.0 + x2) ** (-2.0)

    def density_3d(self, r_pc: jnp.ndarray, *, re_pc: Any) -> jnp.ndarray:
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
torch
        CDF(R) = R^2 / (R^2 + re^2)  ->  R = re * sqrt(u/(1-u)).
        """
        u = jax.random.uniform(key, shape=(n,), minval=0.0, maxval=1.0)
        u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
        re = jnp.asarray(re_pc)
        return re * jnp.sqrt(u / (1.0 - u))


class DMModel(Model):
    required_models: Mapping[str, type[Model]] = {}

    def mass_density_3d(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        raise NotImplementedError

    def enclosed_mass_analytic(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        raise NotImplementedError

    def enclosed_mass_numeric(
        self,
        r_pc: jnp.ndarray,
        *,
        params: Mapping[str, Any],
        n_steps: int = 256,
        t_min: float = 1e-6,
    ) -> jnp.ndarray:
        """Numerically compute enclosed mass via 4π∫ρ(r)r²dr.

        This default path is AD-friendly and avoids special-function gradient issues.
        If `r_t_pc` exists in `params`, radius is truncated at that value.
        """
        r = jnp.asarray(r_pc)
        dtype = jnp.result_type(r)

        if "r_t_pc" in params:
            r_t = jnp.asarray(params["r_t_pc"], dtype=dtype)
            r = jnp.minimum(r.astype(dtype), r_t)
        else:
            r = r.astype(dtype)

        r_pos = jnp.maximum(r, jnp.asarray(0.0, dtype=dtype))
        t = jnp.linspace(jnp.asarray(t_min, dtype=dtype), jnp.asarray(1.0, dtype=dtype), int(n_steps))

        r_grid = r_pos[..., None] * t
        rho = self.mass_density_3d(r_grid, params=params)
        integrand_t = 4.0 * jnp.pi * (r_grid**2) * rho * r_pos[..., None]
        mass = _trapz(integrand_t, t, axis=-1)
        mass = jnp.where(r <= 0.0, jnp.zeros_like(mass), mass)
        return jnp.nan_to_num(mass, nan=0.0, neginf=0.0, posinf=1e12)
    
    @property
    def has_analytic_enclosed_mass(self) -> bool:
        return hasattr(self, "enclosed_mass_analytic") and callable(getattr(self, "enclosed_mass_analytic"))

    def enclosed_mass(self, r_pc: jnp.ndarray, method: str = "analytic", *, params: Mapping[str, Any]) -> jnp.ndarray:
        """Return enclosed mass with selectable backend.

        Default is `numeric` to keep autodiff robust for gradient-based samplers
        when analytic special functions (e.g. betainc) cause AD issues.
        """
        has_analytic = self.has_analytic_enclosed_mass
        if method == "analytic":
            if has_analytic:
                return self.enclosed_mass_analytic(r_pc, params=params)
            else:
                raise NotImplementedError(f"{self.__class__.__name__} does not have an analytic enclosed_mass implementation")
        elif method == "numeric":
            return self.enclosed_mass_numeric(r_pc, params=params)
        else:
            raise ValueError(f"method must be 'analytic' or 'numeric', got {method!r}")


class NFWModel(DMModel):
    required_param_names = ("rs_pc", "rhos_Msunpc3", "r_t_pc")

    def mass_density_3d(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        rs_pc = params["rs_pc"]
        rhos_Msunpc3 = params["rhos_Msunpc3"]

        rs = jnp.asarray(rs_pc)
        rhos = jnp.asarray(rhos_Msunpc3)
        r = jnp.asarray(r_pc)
        x = r / rs
        return rhos / x / (1.0 + x) ** 2

    def enclosed_mass_analytic(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        rs_pc = params["rs_pc"]
        rhos_Msunpc3 = params["rhos_Msunpc3"]
        r_t_pc = params["r_t_pc"]

        rs = jnp.asarray(rs_pc)
        rhos = jnp.asarray(rhos_Msunpc3)
        r_t = jnp.asarray(r_t_pc)

        r = jnp.minimum(jnp.asarray(r_pc), r_t)
        x = r / rs
        # M(r) = 4π ρs rs^3 [ ln(1+x) - x/(1+x) ]
        coeff = 4.0 * jnp.pi * rhos * rs**3
        return coeff * (jnp.log1p(x) - x / (1.0 + x))


class ZhaoModel(DMModel):
    required_param_names = ("rs_pc", "rhos_Msunpc3", "a", "b", "g", "r_t_pc")

    def mass_density_3d(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        rs_pc = params["rs_pc"]
        rhos_Msunpc3 = params["rhos_Msunpc3"]
        a = params["a"]
        b = params["b"]
        g = params["g"]

        rs = jnp.asarray(rs_pc)
        rhos = jnp.asarray(rhos_Msunpc3)
        a_arr = jnp.asarray(a)
        b_arr = jnp.asarray(b)
        g_arr = jnp.asarray(g)
        r = jnp.asarray(r_pc)
        x = r / rs
        return rhos * x ** (-g_arr) * (1.0 + x**a_arr) ** (-(b_arr - g_arr) / a_arr)

    def enclosed_mass_analytic(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        rs_pc = params["rs_pc"]
        rhos_Msunpc3 = params["rhos_Msunpc3"]
        r_t_pc = params["r_t_pc"]
        a = params["a"]
        b = params["b"]
        g = params["g"]

        rs = jnp.asarray(rs_pc)
        rhos = jnp.asarray(rhos_Msunpc3)
        r_t = jnp.asarray(r_t_pc)
        a_arr = jnp.asarray(a)
        b_arr = jnp.asarray(b)
        g_arr = jnp.asarray(g)

        r = jnp.minimum(jnp.asarray(r_pc), r_t)
        x_raw = r / rs
        x = x_raw**a_arr

        argbeta0 = (3.0 - g_arr) / a_arr
        argbeta1 = (b_arr - 3.0) / a_arr
        z = x / (1.0 + x)

        tol = jnp.asarray(1e-7, dtype=jnp.result_type(a_arr, b_arr, g_arr))
        is_nfw_limit = (
            (jnp.abs(a_arr - 1.0) <= tol)
            & (jnp.abs(b_arr - 3.0) <= tol)
            & (jnp.abs(g_arr - 1.0) <= tol)
        )

        argbeta1_safe = jnp.where(is_nfw_limit, jnp.asarray(1.0, dtype=argbeta1.dtype), argbeta1)
        coeff_general = 4.0 * jnp.pi * rs**3 * rhos / a_arr
        mass_general = coeff_general * jsp.beta(argbeta0, argbeta1_safe) * jsp.betainc(argbeta0, argbeta1_safe, z)

        coeff_nfw = 4.0 * jnp.pi * rhos * rs**3
        mass_nfw = coeff_nfw * (jnp.log1p(x_raw) - x_raw / (1.0 + x_raw))

        return jnp.where(is_nfw_limit, mass_nfw, mass_general)


class AnisotropyModel(Model):
    r"""Base interface for anisotropy models used in Jeans LOS integration.

    The note defines

        f(r) = f(r0) * exp(\int_{r0}^{r} 2 beta(t) / t dt)

    and the projected-dispersion kernel formulation through beta(r), f(r), and K(u).
    Subclasses are expected to implement these quantities consistently.
    """

    required_models: Mapping[str, type[Model]] = {}

    def beta(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        """Anisotropy profile beta(r) at 3D radius `r_pc`."""
        raise NotImplementedError

    def f(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        """Auxiliary function f(r) satisfying d ln f / d ln r = 2 beta(r)."""
        raise NotImplementedError

    def kernel(self, u: jnp.ndarray, R_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        r"""Kernel K(u) entering sigma_los^2(R) = 2 * \int du ... K(u)/u."""
        raise NotImplementedError


class ConstantAnisotropyModel(AnisotropyModel):
    r"""Constant anisotropy model with beta(r) = beta_ani.

    For constant beta,

        f(r) = r^{2 beta_ani}

    and K(u) has a closed form involving Gauss hypergeometric function.
    """

    required_param_names = ("beta_ani",)

    def beta(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        """Return constant beta_ani with the same shape as `r_pc`."""
        return jnp.asarray(params["beta_ani"]) * jnp.ones_like(r_pc)

    def f(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        """Return f(r)=r^(2 beta_ani) from the note definition."""
        beta_ani = jnp.asarray(params["beta_ani"])
        r = jnp.asarray(r_pc)
        return r ** (2.0 * beta_ani)

    def kernel(self, u: jnp.ndarray, R_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        r"""LOSVD kernel K(u) for constant anisotropy.

        Uses the transformed hypergeometric representation

            K(u) = sqrt(1-u^{-2}) * ((3/2-beta) * 2F1(1,beta;3/2;1-u^{-2}) - 1/2)

        which is algebraically equivalent to the original form in `model.py` and
        numerically stable for large `u`.

        Parameters
        ----------
        u:
            Dimensionless radius ratio u=r/R (typically u>1).
        R_pc:
            Kept for API compatibility; K(u) is independent of R in this model.
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
        if _HYP2F1_BACKEND == "jax":
            return _constant_kernel_jax_backend(jnp.asarray(u), beta_ani)

        u_arr = jnp.asarray(u)
        dtype = jnp.result_type(u_arr, beta_ani)
        one = jnp.asarray(1.0, dtype=dtype)
        eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        u_safe = jnp.maximum(u_arr.astype(dtype), one + eps)
        u2 = u_safe**2

        hyp_main = _hyp2f1_1_b_3half_from_u_scipy(u_safe, beta_ani)
        hyp_asym = _hyp2f1_1_b_3half_asymptotic_from_u(u_safe, beta_ani, n_terms_regular=4)
        use_asym = ((u_safe > jnp.asarray(20.0, dtype=dtype)) & (beta_ani > 0.0) & (beta_ani < 1.49)) | (~jnp.isfinite(hyp_main))
        hyp_stable = jnp.where(use_asym, hyp_asym, hyp_main)

        pref = jnp.sqrt(1.0 - 1.0 / u2)
        kernel_val = pref * ((1.5 - beta_ani) * hyp_stable - 0.5)
        return jnp.where(u_arr <= 1.0, jnp.zeros_like(kernel_val), kernel_val)


class BaesAnisotropyModel(AnisotropyModel):
    r"""Baes & van Hese anisotropy model with numerical LOS kernel integration.

    Notes
    -----
    This follows the same kernel normalization used in `model.py`:

        sigma_los^2(R) = 2 * \int du [nu(uR)/Sigma(R)] * GM(uR) * K(u)/u

    so `K(u)` itself is computed without an extra prefactor 2 in the inner integral.
    """

    required_param_names = ("beta_0", "beta_inf", "r_a", "eta")

    def beta(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        r"""Baes & van Hese profile

            beta(r) = (beta_0 + beta_inf (r/r_a)^eta) / (1 + (r/r_a)^eta).
        """
        beta_0 = jnp.asarray(params["beta_0"])
        beta_inf = jnp.asarray(params["beta_inf"])
        r_a = jnp.asarray(params["r_a"])
        eta = jnp.asarray(params["eta"])

        x = (jnp.asarray(r_pc) / r_a) ** eta
        return (beta_0 + beta_inf * x) / (1.0 + x)

    def f(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        r"""Return

            f(r)=r^{2 beta_0} (1 + (r/r_a)^eta)^{2(beta_inf-beta_0)/eta}.
        """
        beta_0 = jnp.asarray(params["beta_0"])
        beta_inf = jnp.asarray(params["beta_inf"])
        r_a = jnp.asarray(params["r_a"])
        eta = jnp.asarray(params["eta"])

        r = jnp.asarray(r_pc)
        x = (r / r_a) ** eta
        return r ** (2.0 * beta_0) * (1.0 + x) ** (2.0 * (beta_inf - beta_0) / eta)

    def _log_f(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        """Logarithm of f(r) used for stable ratio evaluation f(s)/f(r)."""
        beta_0 = jnp.asarray(params["beta_0"])
        beta_inf = jnp.asarray(params["beta_inf"])
        r_a = jnp.asarray(params["r_a"])
        eta = jnp.asarray(params["eta"])

        r = jnp.asarray(r_pc)
        x = (r / r_a) ** eta
        return 2.0 * beta_0 * jnp.log(r) + (2.0 * (beta_inf - beta_0) / eta) * jnp.log1p(x)

    def kernel(
        self,
        u: jnp.ndarray,
        R_pc: jnp.ndarray,
        *,
        params: Mapping[str, Any],
        n_kernel: int = 128,
    ) -> jnp.ndarray:
        r"""LOSVD kernel K(u) for general BAES anisotropy via numerical integration.

        Implements the note definition

            K(u_s) = f(Ru_s)/u_s * \int_1^{u_s} du
                     [u/sqrt(u^2-1)] * (1-beta(Ru)/u^2) / f(Ru).

        A fixed-grid JAX-friendly quadrature is used. The change of variables

            u=cosh(s),  s=arccosh(u)

        removes the endpoint singularity at u=1 and keeps the integration interval
        length O(log u), improving accuracy for very large u with fixed n_kernel.
        Internally, f-ratios are computed via log-differences for numerical
        stability in extreme beta regimes.

        Parameters
        ----------
        n_kernel:
            Number of Gauss-Legendre nodes for the inner quadrature.
        """
        return _baes_kernel_jax_backend(
            jnp.asarray(u),
            jnp.asarray(R_pc),
            jnp.asarray(params["beta_0"]),
            jnp.asarray(params["beta_inf"]),
            jnp.asarray(params["r_a"]),
            jnp.asarray(params["eta"]),
            n_kernel=int(n_kernel),
        )


class OsipkovMerrittModel(AnisotropyModel):
    r"""Osipkov-Merritt anisotropy model.

    This corresponds to the BAES special case (beta_0,beta_inf,eta)=(0,1,2):

        beta(r) = r^2 / (r^2 + r_a^2),
        f(r)    = 1 + r^2/r_a^2.
    """

    required_param_names = ("r_a",)

    def beta(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        """Return beta(r)=r^2/(r^2+r_a^2)."""
        r_a = jnp.asarray(params["r_a"])
        r = jnp.asarray(r_pc)
        return r**2 / (r**2 + r_a**2)

    def f(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        """Return f(r)=1+r^2/r_a^2 from the note table."""
        r_a = jnp.asarray(params["r_a"])
        r = jnp.asarray(r_pc)
        return (r_a**2 + r**2) / r_a**2

    def kernel(self, u: jnp.ndarray, R_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        r"""Closed-form K(u) for Osipkov-Merritt anisotropy.

        Uses the analytical expression equivalent to the note/CLUMPY formula,
        with u_a=r_a/R and u=r/R.
        """
        return _osipkov_kernel_jax_backend(
            jnp.asarray(u),
            jnp.asarray(R_pc),
            jnp.asarray(params["r_a"]),
        )


class DSphModel(Model):
    """Minimal Jeans model for sigma_los(R) using a fixed quadrature grid.

    This implementation is intentionally limited:
    - StellarModel: currently assumed to be PlummerModel-compatible interface
    - DMModel: any subclass implementing enclosed_mass(r_pc, params=...)
    - AnisotropyModel: any subclass implementing beta/f/kernel consistently

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
        use_analytic_dm: bool = True,
    ) -> jnp.ndarray:
        r"""Compute projected velocity dispersion sigma_los^2(R).

        With u=r/R, this computes

            sigma_los^2(R) = 2 * \int_1^\infty du
                [nu(uR)/Sigma(R)] [G M(uR)] K(u)/u,

        consistent with the kernel normalization used in `model.py` and the note.
        """
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
        nu3 = stellar.density_3d(r, re_pc=re_pc)
        sig2 = stellar.density_2d(R2d, re_pc=re_pc)
        if use_analytic_dm and dm.has_analytic_enclosed_mass:
            M = dm.enclosed_mass(r, method="analytic", params=params)
        else:
            M = dm.enclosed_mass(r, method="numeric", params=params)

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
    "ZhaoModel",
    "NFWModel",
    "ConstantAnisotropyModel",
    "BaesAnisotropyModel",
    "OsipkovMerrittModel",
    "DSphModel",
]
