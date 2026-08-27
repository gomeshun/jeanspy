"""Private reference/benchmark helpers for spherical Sérsic deprojection.

These helpers are intentionally kept out of the public JeansPy API.  They are
used to validate :class:`jeanspy.model.SersicModel` approximations against
independently implemented literature formulae.

Implemented references
----------------------
* Simonneau & Prada (2004), five-point discrete-ordinate approximation.
* Ciotti, De Deo & Pellegrini (2025), asymptotically matched approximation,
  including the Appendix-A extension for 0.5 < n < 1.  The exact n=0.5 and
  n=1 deprojections are used at those two special indices.

All densities returned here are normalized to unit total luminosity, matching
``SersicModel.density_2d`` and ``SersicModel.density_3d_numerical``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import beta as beta_fn
from scipy.special import gamma, k0


# Simonneau & Prada (2004), Table 1, N_ap = 5, transformed to (0, 1).
_SP04_X = np.array([0.046910, 0.230765, 0.500000, 0.769235, 0.953090])
_SP04_W = np.array([0.118464, 0.239314, 0.284444, 0.239314, 0.118464])


def _as_scalar_or_array(values: np.ndarray, scalar_input: bool):
    if scalar_input:
        return float(values.reshape(-1)[0])
    return values


def sp04_density(r_pc, *, re_pc: float, n: float, b: float):
    """Five-point Simonneau & Prada (2004) Sérsic deprojection.

    Parameters
    ----------
    r_pc : float or array_like
        Three-dimensional radius in parsec.  Must be strictly positive.
    re_pc : float
        Projected half-light radius in parsec.
    n : float
        Sérsic index.  The SP04 transformation is defined for ``n > 1``.
    b : float
        Sérsic ``b_n`` associated with the projected profile.

    Returns
    -------
    float or ndarray
        Unit-total-luminosity three-dimensional density in pc^-3.

    Notes
    -----
    This implements Eqs. (10)--(12) of Simonneau & Prada (2004) with the
    five-point Gaussian quadrature nodes and weights from their Table 1.
    Five points are sufficient at roughly the percent level over the broad
    domain discussed in that paper, and become particularly accurate at the
    high Sérsic indices where Vitral & Mamon (2021) use SP04 in their density
    hybrid.
    """
    if not (n > 1.0):
        raise ValueError(f"SP04 deprojection requires n > 1; got n={n}.")
    if re_pc <= 0:
        raise ValueError(f"re_pc must be positive; got {re_pc}.")

    scalar_input = np.ndim(r_pc) == 0
    r_arr = np.asarray(r_pc, dtype=float)
    if np.any(r_arr <= 0):
        raise ValueError("SP04 deprojection requires strictly positive r_pc.")

    s = np.atleast_1d(r_arr / re_pc)
    lam = (1.0 - _SP04_X**2) ** (-1.0 / (n - 1.0))
    rho_j = _SP04_W * _SP04_X / np.sqrt(
        1.0 - (1.0 - _SP04_X**2) ** (2.0 * n / (n - 1.0))
    )

    # Eq. (10), after substituting the unit-luminosity Sérsic I(0).
    prefactor = b ** (2.0 * n + 1.0) / (
        np.pi**2 * n * (n - 1.0) * gamma(2.0 * n) * re_pc**3
    )
    s_flat = s.reshape(-1)
    summed = np.sum(
        rho_j[:, None]
        * np.exp(-b * lam[:, None] * s_flat[None, :] ** (1.0 / n)),
        axis=0,
    )
    values = prefactor * s_flat ** (1.0 / n - 1.0) * summed
    values = values.reshape(s.shape)
    return _as_scalar_or_array(values, scalar_input)


def _ciotti2025_shape(n: float, b: float):
    """Return ``(c1, c2, alpha, q)`` for the 2025 matched formula.

    The dimensionless density is represented as

    ``c1 exp(-b s^(1/n)) s^alpha / (1 + (c2 s^q)^p)^(1/p)``.
    """
    if n > 1.0:
        B = beta_fn(0.5, (n - 1.0) / (2.0 * n))
        c1 = b ** (2.0 * n + 1.0) * B / (
            4.0 * np.pi**2 * n**2 * gamma(2.0 * n)
        )
        c2 = B * np.sqrt(b / (2.0 * np.pi * n))
        alpha = 1.0 / n - 1.0
        q = 1.0 / (2.0 * n)
        return c1, c2, alpha, q

    if 0.5 < n < 1.0:
        # Appendix A of Ciotti, De Deo & Pellegrini (2025).
        c1 = b ** (3.0 * n) * gamma(1.0 - n) / (
            2.0 * np.pi**2 * n * gamma(2.0 * n)
        )
        c2 = gamma(1.0 - n) * b ** (n - 0.5) * np.sqrt(2.0 * n / np.pi)
        alpha = 0.0
        q = 1.0 - 1.0 / (2.0 * n)
        return c1, c2, alpha, q

    raise ValueError("The matched Ciotti+2025 branch requires 0.5 < n < 1 or n > 1.")


def _ciotti2025_log_density_dimensionless(s, *, n: float, b: float, p: float):
    c1, c2, alpha, q = _ciotti2025_shape(n, b)
    log_s = np.log(s)
    # log[(1 + (c2 s^q)^p)^(1/p)] without overflow.
    log_denominator = np.logaddexp(0.0, p * (np.log(c2) + q * log_s)) / p
    return (
        np.log(c1)
        - b * s ** (1.0 / n)
        + alpha * log_s
        - log_denominator
    )


def _ciotti2025_luminosity(p: float, *, n: float, b: float) -> float:
    """Dimensionless total luminosity of the matched profile."""

    def integrand(log_s):
        s = np.exp(log_s)
        log_value = (
            np.log(4.0 * np.pi)
            + _ciotti2025_log_density_dimensionless(s, n=n, b=b, p=p)
            + 3.0 * log_s
        )
        if log_value < -745.0:
            return 0.0
        return float(np.exp(log_value))

    # The exponential Sérsic tail makes |log s|=40 comfortably sufficient for
    # the n-range benchmarked here while retaining a stable finite interval.
    value, _ = quad(
        integrand,
        -40.0,
        40.0,
        epsrel=2e-10,
        epsabs=1e-12,
        limit=400,
    )
    return value


@lru_cache(maxsize=128)
def _ciotti2025_matching_p_cached(n: float, b: float) -> float:
    if not (n >= 0.55):
        raise ValueError(
            "Ciotti+2025 benchmark is supported for n >= 0.55; n=0.5 is handled "
            "by the exact Gaussian solution."
        )
    if np.isclose(n, 1.0, rtol=0.0, atol=1e-14):
        raise ValueError("n=1 has an exact K0 deprojection and does not require p.")

    # The published Table B.1 solutions lie in these broad brackets.  The
    # low-n branch approaches a large p as n -> 0.5+, hence the wider bracket.
    bracket = (0.01, 200.0) if n < 1.0 else (0.01, 10.0)

    def residual(p):
        return _ciotti2025_luminosity(p, n=n, b=b) - 1.0

    return float(
        brentq(
            residual,
            bracket[0],
            bracket[1],
            xtol=1e-11,
            rtol=1e-11,
            maxiter=200,
        )
    )


def ciotti2025_matching_p(n: float, b: float) -> float:
    """Solve the Ciotti+2025 matching exponent from luminosity conservation.

    The paper fixes the otherwise free matching exponent ``p`` by requiring
    ``4*pi*integral rho(r) r^2 dr = L``.  Solving that condition rather than
    hard-coding Table B.1 provides an independent implementation that can be
    checked against the published tabulated values.
    """
    return _ciotti2025_matching_p_cached(float(n), float(b))


def ciotti2025_density(r_pc, *, re_pc: float, n: float, b: float, p: float | None = None):
    """Ciotti, De Deo & Pellegrini (2025) asymptotically matched density.

    The result is normalized to unit total luminosity.  For ``n=0.5`` and
    ``n=1`` the exact Gaussian and modified-Bessel ``K0`` deprojections are
    returned.  For other supported ``n``, ``p`` is obtained from luminosity
    conservation unless explicitly supplied.

    This helper is for independent benchmarking; it is deliberately not wired
    into ``SersicModel.density_3d`` as a public method yet.
    """
    if re_pc <= 0:
        raise ValueError(f"re_pc must be positive; got {re_pc}.")
    if n < 0.5:
        raise ValueError(f"Ciotti+2025 benchmark requires n >= 0.5; got n={n}.")

    scalar_input = np.ndim(r_pc) == 0
    r_arr = np.asarray(r_pc, dtype=float)
    if np.any(r_arr < 0):
        raise ValueError("r_pc must be non-negative.")
    s = np.atleast_1d(r_arr / re_pc)

    if np.isclose(n, 0.5, rtol=0.0, atol=1e-14):
        values = (b / np.pi) ** 1.5 * np.exp(-b * s**2) / re_pc**3
        return _as_scalar_or_array(values, scalar_input)

    if np.isclose(n, 1.0, rtol=0.0, atol=1e-14):
        values = b**3 / (2.0 * np.pi**2 * re_pc**3) * k0(b * s)
        return _as_scalar_or_array(values, scalar_input)

    if n < 0.55:
        raise ValueError(
            "For 0.5 < n < 0.55 the matched approximation becomes poorly "
            "conditioned; use numerical Abel deprojection instead."
        )

    if p is None:
        p = ciotti2025_matching_p(n, b)
    if p <= 0:
        raise ValueError(f"p must be positive; got {p}.")

    c1, _, alpha, _ = _ciotti2025_shape(n, b)
    values = np.empty_like(s, dtype=float)
    zero = s == 0
    positive = ~zero

    if np.any(positive):
        values[positive] = np.exp(
            _ciotti2025_log_density_dimensionless(
                s[positive], n=n, b=b, p=p
            )
        ) / re_pc**3

    if np.any(zero):
        # Exact inner asymptotics of the matched formula.
        values[zero] = np.inf if n > 1.0 else c1 / re_pc**3

    return _as_scalar_or_array(values, scalar_input)
