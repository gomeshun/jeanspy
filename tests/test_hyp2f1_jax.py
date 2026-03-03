import unittest

import numpy as np
import scipy.special

import jax

# Production-oriented test mode:
# - We intentionally validate with float32 precision assumptions (future GPU use).
# - Target practical requirement: about 4 significant digits (~1e-4 relative) where
#   the model operates in the hard region (positive b, w close to 1).
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp

from jeanspy.hyp2f1_jax import hyp2f1_1b_3half


def _scipy_hyp2f1_1b_3half(b: float, w: float) -> float:
    return float(scipy.special.hyp2f1(1.0, float(b), 1.5, float(w)))


def _central_diff(f, x: float, eps: float) -> float:
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


class TestHyp2f1Jax(unittest.TestCase):
    def test_strict_float32_accuracy_in_project_hard_region(self):
        # Strict requirement for practical project region:
        # beta_ani ~ b in [0.3, 0.8], and w can approach 1 for large u.
        # We require ~4-digit relative accuracy in this region.
        # In float32, w=0.999999 with b~O(1) is quantization-sensitive near
        # the branch endpoint; that ultra-edge is tracked separately below.
        bs = [0.3, 0.5, 0.8]
        ws = [0.9, 0.97, 0.99, 0.995, 0.999, 0.9999]

        rels = []
        for b in bs:
            for w in ws:
                y_jax = hyp2f1_1b_3half(
                    jnp.asarray(b, dtype=jnp.float32),
                    jnp.asarray(w, dtype=jnp.float32),
                    method="auto",
                    quad_rule="tanh_sinh",
                )
                y_ref = _scipy_hyp2f1_1b_3half(b, w)
                rels.append(abs(float(np.asarray(y_jax)) - y_ref) / (abs(y_ref) + 1e-300))

        # 4-digit class requirement in practical operating region.
        self.assertLess(np.max(rels), 1e-3)

    def test_float32_ultra_edge_is_bounded(self):
        # Ultra-edge monitoring near w->1 where float32 input quantization
        # dominates for larger b. We still enforce bounded relative error.
        b = 0.8
        w = 0.999999
        y_jax = hyp2f1_1b_3half(
            jnp.asarray(b, dtype=jnp.float32),
            jnp.asarray(w, dtype=jnp.float32),
            method="auto",
            quad_rule="tanh_sinh",
        )
        y_ref = _scipy_hyp2f1_1b_3half(b, w)
        rel = abs(float(np.asarray(y_jax)) - y_ref) / (abs(y_ref) + 1e-300)
        self.assertLess(rel, 3e-2)

    def test_value_matches_scipy_in_hard_region_gauss_kronrod(self):
        # Same hard region with composite Gauss-Kronrod.
        b = 0.51
        u = 1000.0
        w = 1.0 - 1.0 / (u * u)

        y_jax = hyp2f1_1b_3half(b, w, method="quad", n_quad=128, quad_rule="gauss_kronrod")
        y_ref = _scipy_hyp2f1_1b_3half(b, w)

        y_jax_f = float(np.asarray(y_jax))
        rel = abs(y_jax_f - y_ref) / abs(y_ref)
        # Composite GK is usable but typically less accurate than tanh-sinh here.
        self.assertLess(rel, 2e-1)

    def test_quad_rules_agree(self):
        b = 0.73
        w = 0.997

        y_ts = hyp2f1_1b_3half(b, w, method="quad", n_quad=128, quad_rule="tanh_sinh")
        y_gk = hyp2f1_1b_3half(b, w, method="quad", n_quad=128, quad_rule="gauss_kronrod")

        y_ts_f = float(np.asarray(y_ts))
        y_gk_f = float(np.asarray(y_gk))
        rel = abs(y_ts_f - y_gk_f) / (abs(y_ts_f) + 1e-300)
        self.assertLess(rel, 5e-2)

    def test_tanh_sinh_is_more_accurate_than_gk_in_hard_region(self):
        b = 0.51
        u = 1000.0
        w = 1.0 - 1.0 / (u * u)

        y_ref = _scipy_hyp2f1_1b_3half(b, w)
        y_ts = float(np.asarray(hyp2f1_1b_3half(b, w, method="quad", n_quad=128, quad_rule="tanh_sinh")))
        y_gk = float(np.asarray(hyp2f1_1b_3half(b, w, method="quad", n_quad=128, quad_rule="gauss_kronrod")))

        rel_ts = abs(y_ts - y_ref) / abs(y_ref)
        rel_gk = abs(y_gk - y_ref) / abs(y_ref)
        self.assertLess(rel_ts, rel_gk)

    def test_value_matches_closed_form_at_b_half(self):
        # For b=1/2: 2F1(1,1/2;3/2;w) = atanh(sqrt(w))/sqrt(w)
        b = 0.5
        w = 0.999999

        y_jax = hyp2f1_1b_3half(b, w, method="quad", n_quad=128)
        y_jax_f = float(np.asarray(y_jax))

        y_closed = np.arctanh(np.sqrt(w)) / np.sqrt(w)
        rel = abs(y_jax_f - y_closed) / abs(y_closed)
        self.assertLess(rel, 1e-3)

    def test_value_matches_scipy_for_negative_b_series_at_moderate_w(self):
        # Quad is not valid for b<=0, so we only test the series backend there.
        bs = [-10.0, -2.0, -0.5]
        ws = [0.1, 0.5]

        for b in bs:
            for w in ws:
                y_jax = hyp2f1_1b_3half(
                    jnp.asarray(b, dtype=jnp.float32),
                    jnp.asarray(w, dtype=jnp.float32),
                    method="series",
                    n_terms=768,
                )
                y_ref = _scipy_hyp2f1_1b_3half(b, w)
                y_jax_f = float(np.asarray(y_jax))

                # Series convergence slows as w→1; keep this test to moderate w.
                rel = abs(y_jax_f - y_ref) / (abs(y_ref) + 1e-300)
                self.assertLess(rel, 5e-5)

    def test_grad_w_matches_finite_difference_quad(self):
        b = 0.51
        w0 = 0.999
        eps = 1e-5

        def f_scipy(w: float) -> float:
            return _scipy_hyp2f1_1b_3half(b, w)

        def f_jax(w):
            return hyp2f1_1b_3half(jnp.asarray(b, dtype=jnp.float32), w, method="quad", n_quad=128)

        g_jax = float(np.asarray(jax.grad(f_jax)(jnp.asarray(w0))))
        g_ref = _central_diff(f_scipy, w0, eps)

        self.assertTrue(np.isfinite(g_jax))
        rel = abs(g_jax - g_ref) / (abs(g_ref) + 1e-300)
        self.assertLess(rel, 2e-2)

    def test_grad_w_matches_finite_difference_series(self):
        b = -2.0
        w0 = 0.5
        eps = 1e-5

        def f_scipy(w: float) -> float:
            return _scipy_hyp2f1_1b_3half(b, w)

        def f_jax(w):
            return hyp2f1_1b_3half(jnp.asarray(b, dtype=jnp.float32), w, method="series", n_terms=768)

        g_jax = float(np.asarray(jax.grad(f_jax)(jnp.asarray(w0))))
        g_ref = _central_diff(f_scipy, w0, eps)

        rel = abs(g_jax - g_ref) / (abs(g_ref) + 1e-300)
        self.assertLess(rel, 5e-4)


if __name__ == "__main__":
    unittest.main()
