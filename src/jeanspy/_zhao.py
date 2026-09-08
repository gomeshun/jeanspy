"""Cusp-regularized Zhao mass quadrature, shared by NumPy and JAX."""

from functools import lru_cache
import operator

import numpy as np

PARAM_NAMES = ("rs_pc", "rhos_Msunpc3", "a", "b", "g", "r_t_pc")


@lru_cache(maxsize=16)
def quadrature(n_steps):
    n = operator.index(n_steps)
    if n < 8:
        raise ValueError("n_steps must be an integer >= 8")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (nodes + 1) / 2, weights / 2


def valid_domain(r, params, xp):
    rs, rho, a, b, g, rt = (xp.asarray(params[k]) for k in PARAM_NAMES)
    return (
        xp.isfinite(rs) & (rs > 0) & xp.isfinite(rho) & (rho > 0)
        & xp.isfinite(a) & (a > 0) & xp.isfinite(b)
        & xp.isfinite(g) & (g < 3) & (rt > 0) & ~xp.isnan(rt)
        & (r >= 0) & ~xp.isnan(r) & xp.isfinite(xp.minimum(r, rt))
    )


def enclosed_mass(r_pc, params, *, xp, n_steps=128):
    """Integrate separately below rs (power coordinate) and above rs (log).

    With p=3-g and y=x*u**(4/p), the inner density-volume integrand
    becomes proportional to u**3, even as g approaches 3 from below.
    No central cutoff is required. n_steps is the Gauss order per segment.
    Invalid dynamic inputs return NaN, including under JIT.
    """
    r = xp.asarray(r_pc)
    dtype = xp.result_type(r, xp.asarray(params["rs_pc"]), 1.0)
    u, w = (xp.asarray(v, dtype=dtype) for v in quadrature(operator.index(n_steps)))
    rs, rho, a, b, g, rt = (
        xp.asarray(params[k], dtype=dtype) for k in PARAM_NAMES
    )
    valid = valid_domain(r, params, xp)
    # Safe placeholders keep inactive branches finite at zero/invalid inputs.
    rs = xp.where(rs > 0, rs, 1.0)
    a = xp.where(a > 0, a, 1.0)
    p = xp.where(g < 3, 3 - g, 1.0)
    x = xp.where(valid & (r > 0), xp.minimum(r, rt) / rs, 1.0)
    log_c = xp.log(xp.minimum(x, 1.0))[..., None]
    log_u = xp.log(u)
    q = (b - g) / a
    inner_shape = xp.exp(-q * xp.logaddexp(0.0, a * (log_c + 4 / p * log_u)))
    inner = (4 / p) * xp.exp(p * log_c[..., 0]) * xp.sum(w * u**3 * inner_shape, axis=-1)
    length = xp.log(xp.maximum(x, 1.0))
    t = length[..., None] * u
    outer_shape = xp.exp(p * t - q * xp.logaddexp(0.0, a * t))
    outer = length * xp.sum(w * outer_shape, axis=-1)
    mass = 4 * xp.pi * rho * rs**3 * (inner + outer)
    return xp.where(valid, xp.where(r == 0, 0.0, mass), xp.nan)
