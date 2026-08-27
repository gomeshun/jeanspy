"""Analytic and semi-analytic anisotropy kernels.

This module contains reference implementations of closed-form kernel
representations that are useful for validating the numerical Jeans solvers.
They intentionally live outside the JAX backend so that the special-function
identities can be evaluated with SciPy at double precision.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad_vec


def _appell_f1_1_b1_b2_3half(
    x,
    y,
    b1,
    b2,
    *,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
):
    r"""Evaluate ``F1(1; b1, b2; 3/2; x, y)`` for real ``0 <= x,y < 1``.

    The standard Euler representation of Appell's first hypergeometric
    function simplifies for ``a=1`` and ``c=3/2``.  After the substitution
    ``t = 1-s^2`` it becomes

    .. math::

        F_1(1;b_1,b_2;3/2;x,y)
        = \int_0^1
          (1-x+x s^2)^{-b_1}
          (1-y+y s^2)^{-b_2}\,ds.

    This form has no endpoint square-root singularity.  ``quad_vec`` evaluates
    all broadcast array elements together, which makes this a convenient
    high-accuracy reference evaluator even though SciPy does not currently
    expose Appell ``F1`` directly.
    """

    x_arr, y_arr, b1_arr, b2_arr = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(b1, dtype=float),
        np.asarray(b2, dtype=float),
    )

    if np.any(~np.isfinite(x_arr)) or np.any(~np.isfinite(y_arr)):
        raise ValueError("Appell F1 arguments must be finite.")
    if np.any(x_arr < 0.0) or np.any(x_arr >= 1.0):
        raise ValueError("Require 0 <= x < 1 for the eta=2 kernel representation.")
    if np.any(y_arr < 0.0) or np.any(y_arr >= 1.0):
        raise ValueError("Require 0 <= y < 1 for the eta=2 kernel representation.")

    def integrand(s):
        s2 = s * s
        factor_x = (1.0 - x_arr) + x_arr * s2
        factor_y = (1.0 - y_arr) + y_arr * s2
        return np.power(factor_x, -b1_arr) * np.power(factor_y, -b2_arr)

    value, _ = quad_vec(
        integrand,
        0.0,
        1.0,
        epsabs=float(epsabs),
        epsrel=float(epsrel),
    )
    return np.asarray(value, dtype=float)


def baes_eta2_kernel_appell(
    u,
    R_pc,
    *,
    beta_0,
    beta_inf,
    r_a,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
):
    r"""Baes--van Hese LOS kernel for fixed transition exponent ``eta=2``.

    For

    .. math::

        \beta(r)=\frac{\beta_0+\beta_\infty(r/r_a)^2}
                       {1+(r/r_a)^2},

    define

    .. math::

        p=\beta_0,\qquad q=\beta_\infty-\beta_0,\qquad
        a=\frac{r_a}{R},

    and

    .. math::

        z_1=1-u^{-2},\qquad
        z_2=\frac{u^2-1}{u^2+a^2}.

    The Jeans projection kernel used by :mod:`jeanspy.model_numpyro` is then

    .. math::

        K(u,R)=\sqrt{1-u^{-2}}\left[
        F_1(1;p,q;3/2;z_1,z_2)
        -\frac{p}{u^2}F_1(1;p+1,q;3/2;z_1,z_2)
        -\frac{q}{u^2+a^2}F_1(1;p,q+1;3/2;z_1,z_2)
        \right].

    The expression follows by setting ``t=sqrt(r^2/R^2-1)`` in the inner
    projection integral and using the Appell ``F1`` antiderivative.  All Appell
    arguments lie in ``[0,1)`` for ``u>=1``, ``R>0`` and ``r_a>0``.

    Notes
    -----
    This routine is intended first as a high-accuracy reference implementation.
    It evaluates the Appell functions through their nonsingular Euler integral
    because SciPy/JAX do not provide a native ``F1`` implementation.  The
    closed form itself is independent of that numerical evaluation strategy.
    """

    u_arr, R_arr, beta0_arr, betainf_arr, ra_arr = np.broadcast_arrays(
        np.asarray(u, dtype=float),
        np.asarray(R_pc, dtype=float),
        np.asarray(beta_0, dtype=float),
        np.asarray(beta_inf, dtype=float),
        np.asarray(r_a, dtype=float),
    )

    if np.any(~np.isfinite(u_arr)) or np.any(~np.isfinite(R_arr)):
        raise ValueError("u and R_pc must be finite.")
    if np.any(~np.isfinite(beta0_arr)) or np.any(~np.isfinite(betainf_arr)):
        raise ValueError("beta_0 and beta_inf must be finite.")
    if np.any(~np.isfinite(ra_arr)):
        raise ValueError("r_a must be finite.")
    if np.any(u_arr < 1.0):
        raise ValueError("Require u >= 1 for the LOS kernel.")
    if np.any(R_arr <= 0.0):
        raise ValueError("Require R_pc > 0 for the LOS kernel.")
    if np.any(ra_arr <= 0.0):
        raise ValueError("Require r_a > 0 for the LOS kernel.")

    p = beta0_arr
    q = betainf_arr - beta0_arr
    u2 = u_arr * u_arr
    a2 = (ra_arr / R_arr) ** 2

    z1 = np.maximum(1.0 - 1.0 / u2, 0.0)
    z2 = np.maximum((u2 - 1.0) / (u2 + a2), 0.0)

    # Avoid roundoff producing exactly one for finite but very large u.
    one_minus = np.nextafter(1.0, 0.0)
    z1 = np.minimum(z1, one_minus)
    z2 = np.minimum(z2, one_minus)

    f00 = _appell_f1_1_b1_b2_3half(
        z1, z2, p, q, epsabs=epsabs, epsrel=epsrel
    )
    f10 = _appell_f1_1_b1_b2_3half(
        z1, z2, p + 1.0, q, epsabs=epsabs, epsrel=epsrel
    )
    f01 = _appell_f1_1_b1_b2_3half(
        z1, z2, p, q + 1.0, epsabs=epsabs, epsrel=epsrel
    )

    prefactor = np.sqrt(z1)
    kernel = prefactor * (
        f00 - (p / u2) * f10 - (q / (u2 + a2)) * f01
    )
    return np.where(u_arr == 1.0, 0.0, kernel)


__all__ = ["baes_eta2_kernel_appell"]
