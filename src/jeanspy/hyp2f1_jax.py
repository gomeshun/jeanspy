from __future__ import annotations

from dataclasses import dataclass

from ._jax_env import configure_jax_environment

configure_jax_environment()

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, gammasgn, logsumexp


@dataclass(frozen=True)
class GaussLegendre01:
    x: jnp.ndarray
    w: jnp.ndarray


def _gauss_legendre_01(n: int) -> GaussLegendre01:
    """Gauss-Legendre nodes/weights on [0, 1] as JAX arrays.

    Notes
    -----
    - Nodes/weights are computed once on host via NumPy and then embedded as constants.
    - This is fine for JAX JIT/autodiff because these are constants.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    # Map from [-1, 1] to [0, 1]
    t = 0.5 * (x + 1.0)
    wt = 0.5 * w
    return GaussLegendre01(x=jnp.asarray(t), w=jnp.asarray(wt))


_GAUSS_64 = _gauss_legendre_01(64)
_GAUSS_128 = _gauss_legendre_01(128)


@dataclass(frozen=True)
class TanhSinh01:
    x: jnp.ndarray
    w: jnp.ndarray


def _tanh_sinh_01(*, n: int, h: float) -> TanhSinh01:
    """Tanh-sinh quadrature nodes/weights on [0, 1] as JAX arrays.

    This fixed quadrature handles endpoint singularities well, which are present in
    the Euler integral representation for 2F1 near b≈1/2.

    Parameters
    ----------
    n:
        Half-width in steps (total points = 2*n + 1).
    h:
        Step size in the tanh-sinh parameter domain.
    """
    k = np.arange(-n, n + 1, dtype=np.float64)
    t = h * k
    # Map t -> x in (-1, 1)
    s = np.sinh(t)
    u = 0.5 * np.pi * s
    x = np.tanh(u)
    # dx/dt on (-1,1). Use sech(u)^2 = 1 - tanh(u)^2 to avoid overflow.
    du_dt = 0.5 * np.pi * np.cosh(t)
    dx_dt = (1.0 - x * x) * du_dt
    w = dx_dt * h
    # Map from [-1,1] to [0,1]: y = (x+1)/2, dy = dx/2
    y = 0.5 * (x + 1.0)
    wy = 0.5 * w
    return TanhSinh01(x=jnp.asarray(y), w=jnp.asarray(wy))


# Default rule tuned for float32 robustness near w→1 and b≈1/2 (361 points).
_TANH_SINH_120_H008 = _tanh_sinh_01(n=180, h=0.06)


# 15-point Kronrod extension of 7-point Gauss on [-1, 1].
# Positive abscissae only; we mirror them to build full rules.
_GK15_X_POS = jnp.asarray(
    [
        0.9914553711208126,
        0.9491079123427585,
        0.8648644233597691,
        0.7415311855993945,
        0.5860872354676911,
        0.4058451513773972,
        0.2077849550078985,
    ],
)
_GK15_WK_POS = jnp.asarray(
    [
        0.0229353220105292,
        0.0630920926299785,
        0.1047900103222502,
        0.1406532597155259,
        0.1690047266392679,
        0.1903505780647854,
        0.2044329400752989,
    ],
)
_GK15_WK0 = jnp.asarray(0.2094821410847278)
_GK15_XG_POS = jnp.asarray(
    [
        0.9491079123427585,
        0.7415311855993945,
        0.4058451513773972,
    ],
)
_GK15_WG_POS = jnp.asarray(
    [
        0.1294849661688697,
        0.2797053914892767,
        0.3818300505051189,
    ],
)
_GK15_WG0 = jnp.asarray(0.4179591836734694)


def hyp2f1_1b_d_series(b: jax.Array | jnp.ndarray | float, d: jax.Array | jnp.ndarray | float, w: jax.Array | jnp.ndarray | float, *, n_terms: int) -> jnp.ndarray:
    """Fixed-term series for 2F1(1, b; d; w) with w in [0,1).

    Implemented via a stable recurrence:
        term_0 = 1
        term_{n+1} = term_n * (b+n)/(d+n) * w
        sum = Σ_{n=0}^{N-1} term_n

    This is reverse-mode differentiable (static loop bounds).
    """
    w = jnp.asarray(w)
    b = jnp.asarray(b)
    d = jnp.asarray(d)
    dtype = jnp.result_type(b, d, w)
    w = w.astype(dtype)
    b = b.astype(dtype)
    d = d.astype(dtype)

    term0 = jnp.ones_like(w, dtype=dtype)
    total0 = term0
    if n_terms <= 1:
        return total0

    def body(i, state):
        term, total = state
        i_f = jnp.asarray(i, dtype=dtype)
        term = term * (b + i_f) / (d + i_f) * w
        total = total + term
        return term, total

    _, total = jax.lax.fori_loop(0, n_terms - 1, body, (term0, total0))
    return total


def hyp2f1_1b_3half_series(b: jax.Array | jnp.ndarray | float, w: jax.Array | jnp.ndarray | float, *, n_terms: int = 96) -> jnp.ndarray:
    """Series approximation for 2F1(1, b; 3/2; w), w in [0,1)."""
    return hyp2f1_1b_d_series(b, 1.5, w, n_terms=n_terms)


def hyp2f1_1b_3half_quad(
    b: jax.Array | jnp.ndarray | float,
    w: jax.Array | jnp.ndarray | float,
    *,
    n_points: int = 128,
    quad_rule: str = "tanh_sinh",
) -> jnp.ndarray:
    """Numerically stable evaluation of 2F1(1, b; 3/2; w) for w in [0, 1).

    Uses the Euler integral representation (valid for 0 < b < 3/2):

        2F1(1,b;3/2;w) = Γ(3/2) / (Γ(b) Γ(3/2-b)) * ∫_0^1 dt t^{b-1} (1-t)^{1/2-b} / (1-w t)

    We approximate the integral with a fixed quadrature rule on [0,1]:
    - ``quad_rule='tanh_sinh'``: tanh-sinh nodes (robust near endpoint singularities)
    - ``quad_rule='gauss_kronrod'``: composite 15-point Gauss-Kronrod on uniform panels

    This avoids the catastrophic cancellation that appears in analytic-continuation
    formulas near b≈1/2 and w≈1, and works well in float32.
    """
    b = jnp.asarray(b)
    w = jnp.asarray(w)

    def _broadcast_batch(b_arr: jnp.ndarray, w_arr: jnp.ndarray):
        batch_shape = jnp.broadcast_shapes(b_arr.shape, w_arr.shape)
        b_b = jnp.broadcast_to(b_arr, batch_shape)
        w_b = jnp.broadcast_to(w_arr, batch_shape)
        return b_b, w_b, batch_shape

    def _integral_logsumexp(
        t_node: jnp.ndarray,
        w_node: jnp.ndarray,
        b_arr: jnp.ndarray,
        w_arr: jnp.ndarray,
        dtype_: jnp.dtype,
    ):
        finfo = jnp.finfo(dtype_)
        tiny = jnp.asarray(finfo.tiny, dtype_)
        eps = jnp.asarray(finfo.eps, dtype_)
        n_nodes = t_node.shape[0]

        b_b, w_b, batch_shape = _broadcast_batch(b_arr, w_arr)
        t_b = t_node.astype(dtype_).reshape((n_nodes,) + (1,) * len(batch_shape))
        wt_b = w_node.astype(dtype_).reshape((n_nodes,) + (1,) * len(batch_shape))
        b_exp = b_b.astype(dtype_).reshape((1,) + batch_shape)
        w_exp = w_b.astype(dtype_).reshape((1,) + batch_shape)

        # Drop exact endpoint / zero-weight samples (true zero contribution).
        mask = (wt_b > 0.0) & (t_b > 0.0) & (t_b < 1.0)

        # Keep logs finite even on masked-out points to prevent NaN gradients.
        t_safe = jnp.clip(t_b, tiny, 1.0 - eps)
        wt_safe = jnp.where(wt_b > 0.0, wt_b, 1.0)

        log_weight = (b_exp - 1.0) * jnp.log(t_safe) + (0.5 - b_exp) * jnp.log1p(-t_safe)
        denom_arg = jnp.clip(-w_exp * t_safe, -1.0 + eps, jnp.inf)
        log_denom = jnp.log1p(denom_arg)

        log_terms = jnp.where(mask, jnp.log(wt_safe) + log_weight - log_denom, -jnp.inf)
        return logsumexp(log_terms, axis=0)

    def _quad_tanh_sinh(b_arr: jnp.ndarray, w_arr: jnp.ndarray, dtype_: jnp.dtype):
        quad = _TANH_SINH_120_H008
        t_node = quad.x.astype(dtype_)
        w_node = quad.w.astype(dtype_)
        log_integral = _integral_logsumexp(t_node, w_node, b_arr, w_arr, dtype_)
        return log_integral

    def _quad_gauss_kronrod(b_arr: jnp.ndarray, w_arr: jnp.ndarray, dtype_: jnp.dtype):
        n_panels = max(int(n_points), 1)
        edges = jnp.linspace(jnp.asarray(0.0, dtype_), jnp.asarray(1.0, dtype_), n_panels + 1)
        a = edges[:-1]
        bnd = edges[1:]
        half = 0.5 * (bnd - a)
        mid = 0.5 * (bnd + a)

        xk_pos = _GK15_X_POS.astype(dtype_)
        wk_pos = _GK15_WK_POS.astype(dtype_)
        wg_pos = _GK15_WG_POS.astype(dtype_)

        xk = jnp.concatenate((-xk_pos[::-1], jnp.asarray([0.0], dtype=dtype_), xk_pos), axis=0)
        wk = jnp.concatenate((wk_pos[::-1], jnp.asarray([_GK15_WK0.astype(dtype_)], dtype=dtype_), wk_pos), axis=0)

        xg = jnp.concatenate((-_GK15_XG_POS.astype(dtype_)[::-1], jnp.asarray([0.0], dtype=dtype_), _GK15_XG_POS.astype(dtype_)), axis=0)
        wg = jnp.concatenate((wg_pos[::-1], jnp.asarray([_GK15_WG0.astype(dtype_)], dtype=dtype_), wg_pos), axis=0)

        tk = mid[:, None] + half[:, None] * xk[None, :]
        wk_map = half[:, None] * wk[None, :]

        log_integral_k = _integral_logsumexp(tk.reshape(-1), wk_map.reshape(-1), b_arr, w_arr, dtype_)

        tg = mid[:, None] + half[:, None] * xg[None, :]
        wg_map = half[:, None] * wg[None, :]
        log_integral_g = _integral_logsumexp(tg.reshape(-1), wg_map.reshape(-1), b_arr, w_arr, dtype_)

        integral_k = jnp.exp(log_integral_k)
        integral_g = jnp.exp(log_integral_g)
        return log_integral_k, jnp.abs(integral_k - integral_g)

    dtype = jnp.result_type(b, w)
    b = b.astype(dtype)
    w = w.astype(dtype)

    # prefactor in log-space for stability
    log_pref = gammaln(jnp.asarray(1.5, dtype=dtype)) - gammaln(b) - gammaln(jnp.asarray(1.5, dtype=dtype) - b)

    if quad_rule == "tanh_sinh":
        log_integral = _quad_tanh_sinh(b, w, dtype)
        return jnp.exp(log_pref + log_integral)

    if quad_rule == "gauss_kronrod":
        log_integral, _ = _quad_gauss_kronrod(b, w, dtype)
        return jnp.exp(log_pref + log_integral)

    raise ValueError(f"Unknown quad_rule: {quad_rule!r}")


def hyp2f1_1b_3half_asymptotic(
    b: jax.Array | jnp.ndarray | float,
    w: jax.Array | jnp.ndarray | float,
    *,
    n_terms_regular: int = 3,
    b_half_tol: float = 1e-6,
) -> jnp.ndarray:
    """Asymptotic evaluation of 2F1(1,b;3/2;w) for w close to 1.

    Uses the analytic continuation around ``w=1``:

        2F1(1,b;3/2;w)
          = A(b) * 2F1(1,b;b+1/2;1-w)
            + B(b) * (1-w)^(1/2-b) / sqrt(w)

    where
        A(b) = 1 / (1 - 2b),
        B(b) = Gamma(3/2) * Gamma(b-1/2) / Gamma(b).

    For the regular hypergeometric factor we keep only a few terms in ``1-w``,
    which is accurate when ``w`` is sufficiently close to 1. This continuation is
    valid for negative non-integer ``b`` as well; exact non-positive integers and
    negative half-integers are handled by the ``auto`` selector and should stay on
    the power-series path.
    """
    b_arr = jnp.asarray(b)
    w_arr = jnp.asarray(w)
    dtype = jnp.result_type(b_arr, w_arr)
    b_arr = b_arr.astype(dtype)
    w_arr = w_arr.astype(dtype)

    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)
    finfo = jnp.finfo(dtype)
    tiny = jnp.asarray(finfo.tiny, dtype=dtype)
    eps = jnp.asarray(finfo.eps, dtype=dtype)

    # Keep w in the intended domain [0,1) to avoid log/sqrt singularities at
    # exactly w=1 from float32 quantization.
    w_safe = jnp.clip(w_arr, tiny, one - eps)
    delta = one - w_safe
    sqrt_w = jnp.sqrt(w_safe)

    # Exact b=1/2 branch: 2F1(1,1/2;3/2;w) = atanh(sqrt(w))/sqrt(w).
    val_b_half = half * (jnp.log1p(sqrt_w) - jnp.log1p(-sqrt_w)) / sqrt_w
    is_b_half = jnp.abs(b_arr - half) <= jnp.asarray(b_half_tol, dtype=dtype)

    # Avoid poles of A(b), B(b) at b=1/2 in the general branch.
    b_safe = jnp.where(is_b_half, b_arr + jnp.asarray(b_half_tol, dtype=dtype), b_arr)

    regular_factor = hyp2f1_1b_d_series(
        b_safe,
        b_safe + half,
        delta,
        n_terms=max(int(n_terms_regular), 1),
    )
    a_const = one / (one - two * b_safe)
    regular_term = a_const * regular_factor

    log_b_const = (
        gammaln(jnp.asarray(1.5, dtype=dtype))
        + gammaln(b_safe - half)
        - gammaln(b_safe)
    )
    b_const_sign = gammasgn(b_safe - half) * gammasgn(b_safe)
    singular_term = b_const_sign * jnp.exp(log_b_const + (half - b_safe) * jnp.log(delta)) / sqrt_w
    val_general = regular_term + singular_term
    return jnp.asarray(jnp.where(is_b_half, val_b_half, val_general), dtype=dtype)


def hyp2f1_1b_3half(
    b: jax.Array | jnp.ndarray | float,
    w: jax.Array | jnp.ndarray | float,
    *,
    method: str = "auto",
    n_terms: int = 192,
    w_switch: float = 0.65,
    w_asymptotic: float = 0.6,
    w_asymptotic_negative: float = 0.9,
    b_min_quad: float = 1e-6,
    b_max_quad: float = 1.49,
    b_half_avoid_asym: float = 1e-3,
    b_integer_avoid_asym: float = 1e-8,
    n_quad: int = 128,
    quad_rule: str = "tanh_sinh",
    asym_n_terms: int = 20,
) -> jnp.ndarray:
    """2F1(1,b;3/2;w) specialized helper.

    Parameters
    ----------
    method:
        - "series": fixed-term power series.
        - "quad": Euler integral quadrature (0<b<3/2).
        - "asymptotic": asymptotic expansion around w=1.
        - "auto": uses the asymptotic continuation for positive ``b`` once it is
                  more accurate than quadrature, uses it for negative non-integer
                  ``b`` only very near ``w=1``, and otherwise falls back to
                  quadrature or the terminating/power series.

    n_quad:
        - ``quad_rule='tanh_sinh'``: currently ignored (uses precomputed tanh-sinh rule).
        - ``quad_rule='gauss_kronrod'``: number of uniform panels for composite GK.

    w_asymptotic:
        In ``method='auto'``, switch positive ``b`` values from quadrature to
        asymptotic evaluation for ``w >= w_asymptotic``.

    w_asymptotic_negative:
        In ``method='auto'``, allow negative non-integer ``b`` values onto the
        asymptotic branch only for ``w >= w_asymptotic_negative``.

    b_half_avoid_asym:
        In ``method='auto'``, avoid asymptotic branch for
        ``|b-1/2| < b_half_avoid_asym`` and keep quadrature in this region.

    b_integer_avoid_asym:
        In ``method='auto'``, avoid the asymptotic branch when ``b`` is close to a
        non-positive integer pole or negative half-integer pole of the continuation
        coefficient.

    asym_n_terms:
        Number of terms kept for the regular ``2F1(1,b;b+1/2;1-w)`` factor in the
        asymptotic branch.

    Notes
    -----
    - Intended domain for this project is w in [0,1).
    - The "auto" choice is designed specifically to handle the numerically difficult
      region around b≈1/2 and w≈1.
    """
    b_arr = jnp.asarray(b)
    w_arr = jnp.asarray(w)

    if method == "series":
        return hyp2f1_1b_3half_series(b_arr, w_arr, n_terms=n_terms)

    if method == "quad":
        return hyp2f1_1b_3half_quad(b_arr, w_arr, n_points=n_quad, quad_rule=quad_rule)

    if method == "asymptotic":
        return hyp2f1_1b_3half_asymptotic(b_arr, w_arr, n_terms_regular=asym_n_terms)

    if method != "auto":
        raise ValueError(f"Unknown method: {method!r}")

    dtype = jnp.result_type(b_arr, w_arr)
    zero = jnp.asarray(0.0, dtype=dtype)
    b_half_exact_tol = jnp.asarray(1e-6, dtype=dtype)
    near_b_half = jnp.abs(b_arr - 0.5) <= b_half_avoid_asym
    is_b_half_exact = jnp.abs(b_arr - 0.5) <= b_half_exact_tol
    nearest_integer = jnp.round(b_arr)
    near_nonpositive_integer = (b_arr <= zero) & (jnp.abs(b_arr - nearest_integer) <= b_integer_avoid_asym)
    shifted_nearest_integer = jnp.round(b_arr - 0.5)
    near_negative_half_integer = (b_arr < zero) & (jnp.abs((b_arr - 0.5) - shifted_nearest_integer) <= b_integer_avoid_asym)

    use_asym_positive = (b_arr > b_min_quad) & (w_arr >= w_asymptotic)
    use_asym_negative = (b_arr < zero) & (~near_nonpositive_integer) & (~near_negative_half_integer) & (w_arr >= w_asymptotic_negative)
    avoid_asym = near_b_half & (~is_b_half_exact)
    use_asym = (use_asym_positive | use_asym_negative) & (b_arr < b_max_quad) & (~avoid_asym)
    use_quad = (w_arr > w_switch) & (b_arr > b_min_quad) & (b_arr < b_max_quad)

    # Scalar predicate can use lax.cond directly; array predicate needs elementwise
    # selection. We compute all branches for array inputs and pick via jnp.where.
    if use_quad.ndim == 0:
        def _do_asym(_):
            return hyp2f1_1b_3half_asymptotic(b_arr, w_arr, n_terms_regular=asym_n_terms)

        def _do_quad(_):
            return hyp2f1_1b_3half_quad(b_arr, w_arr, n_points=n_quad, quad_rule=quad_rule)

        def _do_series(_):
            return hyp2f1_1b_3half_series(b_arr, w_arr, n_terms=n_terms)

        def _not_asym(_):
            return jax.lax.cond(use_quad, _do_quad, _do_series, operand=None)

        return jax.lax.cond(use_asym, _do_asym, _not_asym, operand=None)

    asym_val = hyp2f1_1b_3half_asymptotic(b_arr, w_arr, n_terms_regular=asym_n_terms)
    quad_val = hyp2f1_1b_3half_quad(b_arr, w_arr, n_points=n_quad, quad_rule=quad_rule)
    series_val = hyp2f1_1b_3half_series(b_arr, w_arr, n_terms=n_terms)
    return jnp.where(use_asym, asym_val, jnp.where(use_quad, quad_val, series_val))
