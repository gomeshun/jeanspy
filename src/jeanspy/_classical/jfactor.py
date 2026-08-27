"""J-factor geometry helpers used by the classical dark-matter profiles."""

from __future__ import annotations

import numpy as np
from scipy.constants import parsec, physical_constants


kg_eV = 1.0 / physical_constants["electron volt-kilogram relationship"][0]
im_eV = 1.0 / physical_constants["electron volt-inverse meter relationship"][0]
solar_mass_kg = 1.9884e30
C0 = (solar_mass_kg * kg_eV) ** 2 * ((1.0 / parsec) * im_eV) ** 5
C1 = (1e9) ** 2 * (1e2 * im_eV) ** 5
C_J = C0 / C1


def _ullio2016_weight(r_pc, s_pc, t_pc, dist_pc):
    """Return the Ullio & Valli (2016) shell weight ``W(r; s, t)``."""
    try:
        r, s, t, dist = np.broadcast_arrays(
            np.asarray(r_pc, dtype=float),
            np.asarray(s_pc, dtype=float),
            np.asarray(t_pc, dtype=float),
            np.asarray(dist_pc, dtype=float),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Ullio geometry arguments must be real numbers.") from exc

    if (
        np.any(~np.isfinite(r))
        or np.any(~np.isfinite(s))
        or np.any(~np.isfinite(t))
        or np.any(~np.isfinite(dist))
        or np.any(r < 0)
        or np.any(s < 0)
        or np.any(t < s)
        or np.any(t > r)
        or np.any(dist <= r)
    ):
        raise ValueError("Require 0 <= s <= t <= r < dist for the Ullio shell weight.")

    r2_minus_s2 = np.maximum((r - s) * (r + s), 0.0)
    r2_minus_t2 = np.maximum((r - t) * (r + t), 0.0)
    dist2_minus_s2 = np.maximum((dist - s) * (dist + s), 0.0)
    dist2_minus_t2 = np.maximum((dist - t) * (dist + t), 0.0)

    numerator = (t - s) * (t + s)
    denominator = (
        np.sqrt(r2_minus_s2 * dist2_minus_t2)
        + np.sqrt(r2_minus_t2 * dist2_minus_s2)
    )
    argument = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    return np.where(r > 0, (r / dist) * np.arcsinh(argument), 0.0)


def _ullio2016_inner_weight(r_pc, dist_pc):
    """Return ``W(r; 0, r)`` with a small-distance series limit."""
    r = np.asarray(r_pc, dtype=float)
    dist = np.asarray(dist_pc, dtype=float)
    if (
        np.any(~np.isfinite(r))
        or np.any(~np.isfinite(dist))
        or np.any(r < 0)
        or np.any(dist <= r)
    ):
        raise ValueError("Require 0 <= r < dist for the Ullio shell weight.")

    q = r / dist
    q2 = q * q
    series = q2 * (1.0 + q2 / 3.0 + q2 * q2 / 5.0 + q2 * q2 * q2 / 7.0)
    return np.where(q2 < 1.0e-8, series, q * np.arctanh(q))


__all__ = ["C_J"]
