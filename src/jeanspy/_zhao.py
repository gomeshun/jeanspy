"""Cusp-regularized Zhao mass quadrature, shared by NumPy and JAX."""

import operator

import numpy as np

from .mass_grid import SphericalMassGrid, _unit_rule

PARAM_NAMES = ("rs_pc", "rhos_Msunpc3", "alpha", "beta", "gamma", "r_t_pc")


def quadrature(n_steps):
    n = operator.index(n_steps)
    if n < 8:
        raise ValueError("n_steps must be an integer >= 8")
    return _unit_rule(n)


def valid_domain(r, params, xp):
    rs, rho, alpha, beta, gamma, rt = (xp.asarray(params[k]) for k in PARAM_NAMES)
    return (
        xp.isfinite(rs) & (rs > 0) & xp.isfinite(rho) & (rho > 0)
        & xp.isfinite(alpha) & (alpha > 0) & xp.isfinite(beta)
        & xp.isfinite(gamma) & (gamma < 3) & (rt > 0) & ~xp.isnan(rt)
        & (r >= 0) & ~xp.isnan(r) & xp.isfinite(xp.minimum(r, rt))
    )


def _mass_parameters(r, params, xp):
    # The numerical mass API chooses precision from radius and scale radius.
    dtype = xp.result_type(r, xp.asarray(params["rs_pc"]), 1.0)
    return tuple(xp.asarray(params[name], dtype=dtype) for name in PARAM_NAMES)


def enclosed_mass(r_pc, params, *, xp, n_steps=128):
    """Integrate separately below rs (power coordinate) and above rs (log).

    With p=3-gamma and y=x*u**(4/p), the inner density-volume integrand
    becomes proportional to u**3, even as gamma approaches 3 from below.
    No central cutoff is required. n_steps is the Gauss order per segment.
    Below rs, the mass prefactor separates the physical radius from rs to
    avoid cancellation in the scale derivative for weakly varying cores.
    Invalid dynamic inputs return NaN, including under JIT.
    """
    r = xp.asarray(r_pc)
    rs, rho, alpha, beta, gamma, rt = _mass_parameters(r, params, xp)
    u, w = (xp.asarray(v, dtype=rs.dtype) for v in quadrature(operator.index(n_steps)))
    valid = valid_domain(r, params, xp)
    # Safe placeholders keep inactive branches finite at zero/invalid inputs.
    rs = xp.where(rs > 0, rs, 1.0)
    alpha = xp.where(alpha > 0, alpha, 1.0)
    p = xp.where(gamma < 3, 3 - gamma, 1.0)
    radius = xp.where(valid & (r > 0), xp.minimum(r, rt), rs)
    x = radius / rs
    log_c = xp.log(xp.minimum(x, 1.0))[..., None]
    log_u = xp.log(u)
    q = (beta - gamma) / alpha
    inner_shape = xp.exp(-q * xp.logaddexp(0.0, alpha * (log_c + 4 / p * log_u)))
    inner = (4 / p) * xp.exp(p * log_c[..., 0]) * xp.sum(w * u**3 * inner_shape, axis=-1)
    length = xp.log(xp.maximum(x, 1.0))
    t = length[..., None] * u
    outer_shape = xp.exp(p * t - q * xp.logaddexp(0.0, alpha * t))
    outer = length * xp.sum(w * outer_shape, axis=-1)
    mass = 4 * xp.pi * rho * rs**3 * (inner + outer)
    # Below rs, combine rs**3 * (r/rs)**(3-gamma) before AD. Otherwise
    # the scale-radius derivative subtracts two order-one contributions
    # even for a core, whose true derivative can be far below roundoff.
    # Bound the inactive branch too: r can be much larger than rs.
    inner_radius = xp.minimum(radius, rs)
    # Keep the combined prefactor in log space: separate powers can overflow
    # and underflow even when their product is representable for steep cusps.
    inner_amplitude = xp.exp(p * xp.log(inner_radius) + gamma * xp.log(rs))
    inner_mass = (
        4 * xp.pi * rho * (4 / p) * inner_amplitude
        * xp.sum(w * u**3 * inner_shape, axis=-1)
    )
    mass = xp.where(x < 1.0, inner_mass, mass)
    return xp.where(valid, xp.where(r == 0, 0.0, mass), xp.nan)


def cumulative_mass(r_pc, params, *, xp, n_steps=128):
    """Use a cusp-regularized anchor and integrate adjacent physical shells."""
    r = xp.asarray(r_pc)
    if r.size < 2:
        return enclosed_mass(r, params, xp=xp, n_steps=n_steps)
    valid = valid_domain(r, params, xp)
    rs, rho, alpha, beta, gamma, rt = _mass_parameters(r, params, xp)
    positive = valid & (r > 0)
    cut = xp.minimum(xp.asarray(r, dtype=rs.dtype), rt)
    first = xp.min(xp.where(positive, cut, xp.inf))
    safe = xp.where(positive, cut, xp.where(xp.isfinite(first), first, 1.))
    grid = SphericalMassGrid(safe, xp=xp)

    def cumulative():
        log_rs = xp.log(xp.where(rs > 0, rs, 1.))
        log_rho = xp.log(xp.where(rho > 0, rho, 1.))
        safe_alpha = xp.where(alpha > 0, alpha, 1.)
        log_r = grid.log_nodes
        transition = xp.logaddexp(0., safe_alpha * (log_r - log_rs))
        log_shell = (xp.log(xp.asarray(4*xp.pi, dtype=rs.dtype)) + log_rho
                     + (3-gamma)*log_r + gamma*log_rs - (beta-gamma)/safe_alpha*transition)
        anchor = enclosed_mass(grid.first_radius, params, xp=xp, n_steps=n_steps)
        mass = grid.integrate(anchor, xp.exp(log_shell))
        return xp.where(valid, xp.where(r == 0, 0., mass), xp.nan)

    def pointwise():
        return enclosed_mass(r, params, xp=xp, n_steps=n_steps)

    # Bound variation across an eight-node panel in both the power-law
    # envelope and the transition. Sparse or sharply broken grids can still
    # use the regularized independent integrations.
    variation = xp.maximum(alpha, xp.maximum(xp.abs(3-beta), xp.abs(3-gamma)))
    dense = grid.max_log_step * xp.maximum(1., variation) <= 2.
    if xp is np:
        return cumulative() if dense else pointwise()
    from jax import lax
    return lax.cond(dense, cumulative, pointwise)
