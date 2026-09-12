"""Shared physical parameter schema; operations support NumPy and JAX NumPy."""

DEFAULTS = {"Q": 1., "alpha": 1., "beta": 3., "gamma": 1.,
            "beta_z": 0., "inclination": 1.5707963267948966}
REQUIRED = {"re_pc", "rs_pc", "rhos_Msunpc3"}


def resolve_params(params, xp):
    """Return safe finite parameters and a scalar validity mask.

    Invalid traced proposals use a benign evaluation point and are rejected by
    the caller. This prevents NaN gradients from invalid inactive branches.
    """
    missing = REQUIRED - params.keys()
    if missing:
        raise ValueError(f"Missing axisymmetric parameters: {sorted(missing)}")
    if ("q" in params) == ("q_projected" in params):
        raise ValueError("Supply exactly one of q and q_projected")
    p = {k: xp.asarray(params.get(k, v), dtype=float) for k, v in DEFAULTS.items()}
    p.update({k: xp.asarray(params[k], dtype=float) for k in REQUIRED})
    shape = xp.asarray(params.get("q", params.get("q_projected")), dtype=float)
    if shape.ndim or any(v.ndim for v in p.values()):
        raise ValueError("Physical parameters must be scalars; use vmap for batches")
    valid = xp.all(xp.stack([xp.isfinite(v) for v in p.values()])) & xp.isfinite(shape)
    for k in ("re_pc", "rs_pc", "rhos_Msunpc3", "Q", "alpha"):
        valid = valid & (p[k] > 0)
    valid = (valid & (p["beta"] > 2) & (p["gamma"] >= 0) & (p["gamma"] < 2)
             & (p["beta_z"] < 1) & (p["inclination"] >= 0)
             & (p["inclination"] <= xp.pi/2) & (shape > 0))
    if "q_projected" in params:
        ci, si = xp.cos(p["inclination"]), xp.sin(p["inclination"])
        q2 = (shape**2-ci**2)/xp.where(si > 0, si**2, 1.)
        valid = valid & (shape <= 1) & (p["inclination"] > 0) & (q2 > 0)
        p["q"] = xp.sqrt(xp.where(xp.isfinite(q2) & (q2 > 0), q2, 1.))
    else:
        p["q"] = shape
    safe = {**DEFAULTS, "re_pc": 300., "rs_pc": 500., "rhos_Msunpc3": .1, "q": 1.}
    return {k: xp.where(valid, v, safe[k]) for k, v in p.items()}, valid
