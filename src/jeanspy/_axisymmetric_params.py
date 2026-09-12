"""Shared physical parameter schema; operations support NumPy and JAX NumPy."""

DEFAULTS = {"Q": 1., "alpha": 1., "beta": 3., "gamma": 1.,
            "beta_z": 0., "inclination": 1.5707963267948966,
            "r_t_pc": float("inf")}
REQUIRED = {"re_pc", "rs_pc", "rhos_Msunpc3"}
SUPPORTED = REQUIRED | DEFAULTS.keys() | {"q", "q_projected", "vmem_kms"}


class InvalidAxisymmetricModelError(ValueError):
    """A physical proposal or its numerical Jeans moments are inadmissible."""


def validate_param_names(names):
    """Check the physical schema without evaluating parameter values."""
    names = set(names)
    unknown = names - SUPPORTED
    if unknown:
        raise ValueError(f"Unknown axisymmetric parameters: {sorted(unknown)}")
    missing = REQUIRED - names
    if missing:
        raise ValueError(f"Missing axisymmetric parameters: {sorted(missing)}")
    if ("q" in names) == ("q_projected" in names):
        raise ValueError("Supply exactly one of q and q_projected")


def resolve_params(params, xp):
    """Return safe finite parameters and a scalar validity mask.

    Invalid traced proposals use a benign evaluation point and are rejected by
    the caller. This prevents NaN gradients from invalid inactive branches.
    """
    validate_param_names(params)
    p = {k: xp.asarray(params.get(k, v), dtype=float) for k, v in DEFAULTS.items()}
    p.update({k: xp.asarray(params[k], dtype=float) for k in REQUIRED})
    shape = xp.asarray(params.get("q", params.get("q_projected")), dtype=float)
    if shape.ndim or any(v.ndim for v in p.values()):
        raise ValueError("Physical parameters must be scalars; use vmap for batches")
    valid = xp.all(xp.stack([xp.isfinite(v) for k, v in p.items() if k != "r_t_pc"])) & xp.isfinite(shape)
    valid = valid & (p["r_t_pc"] > 0) & ~xp.isnan(p["r_t_pc"])
    for k in ("re_pc", "rs_pc", "rhos_Msunpc3", "Q", "alpha"):
        valid = valid & (p[k] > 0)
    valid = (valid & (p["beta"] > 2) & (p["gamma"] >= 0) & (p["gamma"] < 2)
             & (p["beta_z"] < 1) & (p["inclination"] >= 0)
             & (p["inclination"] <= xp.pi/2) & (shape > 0))
    if "q_projected" in params:
        si = xp.sin(p["inclination"])
        q2 = 1-(1-shape)*(1+shape)/xp.where(si > 0, si**2, 1.)
        valid = valid & (shape <= 1) & (p["inclination"] > 0) & (q2 > 0)
        p["q"] = xp.sqrt(xp.where(xp.isfinite(q2) & (q2 > 0), q2, 1.))
    else:
        p["q"] = shape
    safe = {**DEFAULTS, "re_pc": 300., "rs_pc": 500., "rhos_Msunpc3": .1, "q": 1.}
    return {k: xp.where(valid, v, safe[k]) for k, v in p.items()}, valid


def force_limit(R, z, Q, r_t_pc, xp):
    """Upper homoeoidal node and its R derivative for an ellipsoidal cutoff.

    The density contributes only where m(t) <= r_t. Outside the ellipsoid the
    moving upper endpoint also contributes to d(Phi_z)/dR; omitting it violates
    the radial Jeans equation. Interior/infinite-cutoff evaluation is benign.
    """
    rt = xp.where(xp.isfinite(r_t_pc), r_t_pc, 1.)
    outside = xp.isfinite(r_t_pc) & (R*R+(z/Q)**2 > rt*rt)
    # Place inactive inputs on the polar boundary to avoid zero denominators.
    rr, zz = xp.where(outside, R, 0.), xp.where(outside, z, Q*rt)
    delta = Q*Q-1
    b = rr*rr+zz*zz-delta*rt*rt
    root = xp.sqrt(b*b+4*delta*rr*rr*rt*rt)
    # Rationalized quadratic root, with a second form when b < 0.
    positive_denom = xp.where(b >= 0, b+root, 1.)
    other_denom = xp.where(b < 0, 2*delta*rr*rr, 1.)
    t2 = xp.where(b >= 0, 2*rt*rt/positive_denom, (root-b)/other_denom)
    t = xp.sqrt(xp.maximum(t2, 0.))
    d2 = 1+delta*t2
    dt = -rr*t/(rr*rr+zz*zz/d2**2)
    return xp.where(outside, t, 1.), xp.where(outside, dt, 0.)
