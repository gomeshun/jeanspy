"""Specialized Baes--van Hese anisotropy kernel for ``eta = 2``.

For the Baes--van Hese profile

    beta(r) = [beta_0 + beta_inf (r/r_a)^2] / [1 + (r/r_a)^2],

the line-of-sight Jeans kernel can be reduced analytically to Appell's
hypergeometric function F1.  This module keeps that reduction separate from
the generic numerical BAES implementation so that it can be validated before
being promoted to the default runtime path.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Any, Mapping

import numpy as np

# Importing model_numpyro first ensures the repository's JAX environment
# configuration is applied before JAX is imported here.
from .model_numpyro import BaesAnisotropyModel

import jax
import jax.numpy as jnp


DEFAULT_BAES_ETA2_KERNEL_N_QUAD = 96


@lru_cache(maxsize=16)
def _gauss_legendre_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss--Legendre nodes and weights on [0, 1]."""
    n = max(int(n), 8)
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (0.5 * (nodes + 1.0), 0.5 * weights)


@partial(jax.jit, static_argnames=("n_kernel",))
def baes_eta2_kernel_jax(
    u: jnp.ndarray,
    R_pc: jnp.ndarray,
    beta_0: jnp.ndarray,
    beta_inf: jnp.ndarray,
    r_a: jnp.ndarray,
    *,
    n_kernel: int = DEFAULT_BAES_ETA2_KERNEL_N_QUAD,
) -> jnp.ndarray:
    r"""Evaluate the ``eta=2`` BAES kernel from its Appell-F1 representation.

    Write

        p = beta_0,
        q = beta_inf - beta_0,
        a = r_a / R,
        z1 = 1 - u^{-2},
        z2 = (u^2 - 1) / (u^2 + a^2).

    The exact closed form is

        K = sqrt(z1) [
            F1(1;p,q;3/2;z1,z2)
            - p/u^2 F1(1;p+1,q;3/2;z1,z2)
            - q/(u^2+a^2) F1(1;p,q+1;3/2;z1,z2)
        ].

    JAX does not currently provide Appell F1 directly.  We therefore evaluate
    the *Euler representation of this closed form* on a fixed Gauss--Legendre
    grid.  For a=1 in the first Appell argument,

        F1(1;p,q;3/2;z1,z2)
          = integral_0^1 dy
              (1-z1+z1 y^2)^(-p)
              (1-z2+z2 y^2)^(-q).

    Combining the three contiguous F1 terms before quadrature avoids
    cancellation between separately evaluated special functions.  The interval
    is fixed and the endpoint square-root singularity of the original Jeans
    kernel is absent.

    This routine is intended as the JAX-friendly evaluator of the analytic
    reduction.  ``baes_eta2_kernel_appell_reference`` below provides an
    independent high-precision Appell-F1 reference for validation.
    """
    u_arr, r_arr = jnp.broadcast_arrays(jnp.asarray(u), jnp.asarray(R_pc))
    dtype = jnp.result_type(u_arr, r_arr, beta_0, beta_inf, r_a)

    one = jnp.asarray(1.0, dtype=dtype)
    eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)

    u_in = u_arr.astype(dtype)
    u_safe = jnp.maximum(u_in, one + eps)
    r_safe = jnp.maximum(r_arr.astype(dtype), tiny)
    p = jnp.asarray(beta_0, dtype=dtype)
    q = jnp.asarray(beta_inf, dtype=dtype) - p
    r_a_safe = jnp.maximum(jnp.asarray(r_a, dtype=dtype), tiny)

    u2 = u_safe * u_safe
    a2 = (r_a_safe / r_safe) ** 2
    c = one + a2

    # Keep the small complements explicitly instead of forming z=1-delta and
    # later subtracting again.  This avoids float32 loss of significance when
    # u is large.
    delta1 = jnp.clip(one / u2, tiny, one)
    inv_u2_plus_a2 = one / (u2 + a2)
    delta2 = jnp.clip(c * inv_u2_plus_a2, tiny, one)
    z1 = jnp.maximum(one - delta1, jnp.asarray(0.0, dtype=dtype))
    z2 = jnp.maximum(one - delta2, jnp.asarray(0.0, dtype=dtype))

    nodes_np, weights_np = _gauss_legendre_01(int(n_kernel))
    y = jnp.asarray(nodes_np, dtype=dtype)
    weights = jnp.asarray(weights_np, dtype=dtype)
    y2 = y * y

    A = delta1[..., None] + z1[..., None] * y2
    B = delta2[..., None] + z2[..., None] * y2

    log_base = -p * jnp.log(A) - q * jnp.log(B)
    log_base = jnp.clip(log_base, min=-80.0, max=80.0)
    base = jnp.exp(log_base)

    correction = (
        one
        - p * delta1[..., None] / A
        - q * inv_u2_plus_a2[..., None] / B
    )
    kernel_val = jnp.sqrt(z1) * jnp.sum(weights * base * correction, axis=-1)

    kernel_val = jnp.where(u_in <= one, jnp.zeros_like(kernel_val), kernel_val)
    return jnp.nan_to_num(kernel_val, nan=0.0, neginf=0.0, posinf=1e12)


def baes_eta2_kernel_appell_reference(
    u: Any,
    R_pc: Any,
    beta_0: float,
    beta_inf: float,
    r_a: float,
    *,
    dps: int = 40,
) -> np.ndarray:
    r"""High-precision reference evaluation using ``mpmath.appellf1``.

    This function is deliberately a reference path rather than a production
    hot path.  ``mpmath`` is a development dependency of jeanspy and is loaded
    lazily here.  It evaluates the exact three-term Appell-F1 expression quoted
    in :func:`baes_eta2_kernel_jax`.
    """
    try:
        import mpmath as mp
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        raise ImportError(
            "baes_eta2_kernel_appell_reference requires the jeanspy 'dev' dependencies"
        ) from exc

    u_arr, r_arr = np.broadcast_arrays(
        np.asarray(u, dtype=np.float64), np.asarray(R_pc, dtype=np.float64)
    )
    out = np.empty_like(u_arr, dtype=np.float64)

    with mp.workdps(int(dps)):
        p = mp.mpf(float(beta_0))
        q = mp.mpf(float(beta_inf - beta_0))
        r_a_mp = mp.mpf(float(r_a))
        c32 = mp.mpf("1.5")

        for index in np.ndindex(u_arr.shape):
            u_i = mp.mpf(float(u_arr[index]))
            R_i = mp.mpf(float(r_arr[index]))
            if u_i <= 1:
                out[index] = 0.0
                continue
            if R_i <= 0 or r_a_mp <= 0:
                raise ValueError("Require R_pc > 0 and r_a > 0")

            a = r_a_mp / R_i
            u2 = u_i * u_i
            z1 = 1 - 1 / u2
            z2 = (u2 - 1) / (u2 + a * a)

            def f1(pp: Any, qq: Any) -> Any:
                return mp.appellf1(1, pp, qq, c32, z1, z2)

            value = mp.sqrt(z1) * (
                f1(p, q)
                - (p / u2) * f1(p + 1, q)
                - (q / (u2 + a * a)) * f1(p, q + 1)
            )
            out[index] = float(value)

    return out


class BaesEta2AnisotropyModel(BaesAnisotropyModel):
    r"""Baes--van Hese anisotropy with ``eta`` fixed to 2.

    Fixing ``eta=2`` gives

        beta(r) = [beta_0 + beta_inf (r/r_a)^2] / [1 + (r/r_a)^2],

    and admits the Appell-F1 LOS kernel implemented above.  The class is a
    subclass of the generic NumPyro ``BaesAnisotropyModel`` so existing
    ``DSphModel`` kernel plumbing (including ``n_kernel`` forwarding) works
    unchanged.
    """

    required_param_names = ("beta_0", "beta_inf", "r_a")

    def beta(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        beta_0 = jnp.asarray(params["beta_0"])
        beta_inf = jnp.asarray(params["beta_inf"])
        r_a = jnp.asarray(params["r_a"])
        x = (jnp.asarray(r_pc) / r_a) ** 2
        return (beta_0 + beta_inf * x) / (1.0 + x)

    def f(self, r_pc: jnp.ndarray, *, params: Mapping[str, Any]) -> jnp.ndarray:
        beta_0 = jnp.asarray(params["beta_0"])
        beta_inf = jnp.asarray(params["beta_inf"])
        r_a = jnp.asarray(params["r_a"])
        r = jnp.asarray(r_pc)
        x = (r / r_a) ** 2
        return r ** (2.0 * beta_0) * (1.0 + x) ** (beta_inf - beta_0)

    def kernel(
        self,
        u: jnp.ndarray,
        R_pc: jnp.ndarray,
        *,
        params: Mapping[str, Any],
        n_kernel: int = DEFAULT_BAES_ETA2_KERNEL_N_QUAD,
    ) -> jnp.ndarray:
        return baes_eta2_kernel_jax(
            jnp.asarray(u),
            jnp.asarray(R_pc),
            jnp.asarray(params["beta_0"]),
            jnp.asarray(params["beta_inf"]),
            jnp.asarray(params["r_a"]),
            n_kernel=int(n_kernel),
        )


__all__ = [
    "BaesEta2AnisotropyModel",
    "DEFAULT_BAES_ETA2_KERNEL_N_QUAD",
    "baes_eta2_kernel_appell_reference",
    "baes_eta2_kernel_jax",
]
