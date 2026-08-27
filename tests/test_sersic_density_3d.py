"""Tests for the public spherical Sérsic deprojection API."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma, k0

from jeanspy._sersic_deprojection import sp04_density
from jeanspy.model import SersicModel


RE_PC = 100.0


def make_model(n, deprojection_method="numerical"):
    """Construct a model with a method suitable for wide-radius tests."""
    return SersicModel(re_pc=RE_PC, n=n, deprojection_method=deprojection_method)


def _radii(rmin=0.01, rmax=10.0, num=20):
    return np.logspace(np.log10(rmin), np.log10(rmax), num) * RE_PC


def exact_rho_n05(r_pc, re_pc):
    """Exact n=0.5 Gaussian deprojection."""
    m = make_model(0.5)
    b = m.b
    norm2d = m.norm
    sigma2 = re_pc**2 / (2 * b)
    A = 1.0 / norm2d
    return A / np.sqrt(2 * np.pi * sigma2) * np.exp(-r_pc**2 / (2 * sigma2))


def exact_rho_n1(r_pc, re_pc):
    """Exact n=1 exponential deprojection."""
    m = make_model(1.0)
    return (m.b / re_pc) / (np.pi * m.norm) * k0(m.b * r_pc / re_pc)


# ---------------------------------------------------------------------------
# Numerical Abel reference
# ---------------------------------------------------------------------------


def test_numerical_n05_vs_exact():
    m = make_model(0.5)
    r_arr = _radii(0.05, 3.0, 20)
    rho_num = m.density_3d_numerical(r_arr)
    rho_exact = exact_rho_n05(r_arr, RE_PC)
    valid = rho_exact > rho_exact.max() * 1e-8
    rel_err = np.abs(rho_num[valid] / rho_exact[valid] - 1.0)
    assert np.all(rel_err < 1e-4), f"max rel err = {rel_err.max():.2e}"


def test_numerical_n1_vs_exact():
    m = make_model(1.0)
    r_arr = _radii(0.05, 5.0, 25)
    rho_num = m.density_3d_numerical(r_arr)
    rho_exact = exact_rho_n1(r_arr, RE_PC)
    valid = rho_exact > rho_exact.max() * 1e-8
    rel_err = np.abs(rho_num[valid] / rho_exact[valid] - 1.0)
    assert np.all(rel_err < 1e-4), f"max rel err = {rel_err.max():.2e}"


@pytest.mark.parametrize("n", [0.5, 0.75])
def test_numerical_center_n_lt_1_uses_analytic_limit(n):
    m = make_model(n)
    expected = (
        m.b ** (3 * n)
        * gamma(1 - n)
        / (2 * np.pi**2 * n * gamma(2 * n) * RE_PC**3)
    )
    assert m.density_3d_numerical(0.0) == pytest.approx(expected, rel=5e-13)


@pytest.mark.parametrize("n", [1.0, 2.0, 4.0])
def test_numerical_center_diverges_for_n_ge_1(n):
    m = make_model(n)
    assert np.isposinf(m.density_3d_numerical(0.0))


def test_numerical_positive_infinity_is_zero():
    m = make_model(2.0)
    assert m.density_3d_numerical(np.inf) == 0.0


def test_density_3d_numerical_negative_raises():
    m = make_model(2.0)
    with pytest.raises(ValueError, match="non-negative"):
        m.density_3d_numerical(-1.0)


def test_density_3d_numerical_nan_raises():
    m = make_model(2.0)
    with pytest.raises(ValueError, match="NaN"):
        m.density_3d_numerical(np.nan)


@pytest.mark.parametrize("n", [0.5, 1.0, 2.0, 4.0])
def test_numerical_normalization(n):
    """The numerical deprojection should preserve unit total luminosity."""
    m = make_model(n)

    def integrand(r):
        return 4 * np.pi * r**2 * m.density_3d_numerical(r)

    result, err = quad(
        integrand,
        RE_PC * 1e-3,
        RE_PC * 30,
        limit=200,
        epsrel=1e-4,
        epsabs=0,
    )
    assert 0.90 < result < 1.02, (
        f"n={n}: normalization integral = {result:.4f} ± {err:.2e}"
    )


# ---------------------------------------------------------------------------
# Approximation accuracy and validity domains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1.0, 2.0, 4.0, 8.0])
def test_approx_vs_numerical_near_re(n):
    m = make_model(n)
    r_arr = np.array([0.5, 1.0, 2.0]) * RE_PC
    rel_err = np.abs(m.density_3d_LGM(r_arr) / m.density_3d_numerical(r_arr) - 1.0)
    assert np.all(rel_err < 0.05)


def test_approx_vs_numerical_low_n_near_re():
    m = make_model(0.6)
    r_arr = np.array([0.5, 1.0, 2.0]) * RE_PC
    rel_err = np.abs(m.density_3d_LGM(r_arr) / m.density_3d_numerical(r_arr) - 1.0)
    assert np.all(rel_err < 0.10)


@pytest.mark.parametrize("n", [1.0, 2.0, 4.0])
def test_vm20_vs_numerical_near_re(n):
    m = make_model(n)
    r_arr = np.array([0.1, 0.5, 1.0, 2.0, 5.0]) * RE_PC
    rel_err = np.abs(m.density_3d_VM20(r_arr) / m.density_3d_numerical(r_arr) - 1.0)
    assert np.all(rel_err < 0.05)


def test_vm20_low_n_accuracy():
    m = make_model(0.6)
    r_arr = np.array([0.01, 0.05, 0.1, 0.5, 1.0]) * RE_PC
    rho_num = m.density_3d_numerical(r_arr)
    rel_vm20 = np.abs(m.density_3d_VM20(r_arr) / rho_num - 1.0)
    rel_lgm = np.abs(m.density_3d_LGM(r_arr) / rho_num - 1.0)
    assert rel_vm20.mean() <= rel_lgm.mean() * 2.0


def test_vm20_out_of_range_n():
    m = SersicModel(re_pc=RE_PC, n=0.3)
    with pytest.raises(ValueError, match="0.5"):
        m.density_3d_VM20(RE_PC)


def test_vm20_out_of_range_r():
    m = make_model(2.0)
    with pytest.raises(ValueError, match="r/R_e"):
        m.density_3d_VM20(RE_PC * 1e-4)


@pytest.mark.parametrize("n", [0.5, 0.75, 1.0, 2.0])
def test_vm20bis_vs_numerical(n):
    """Official VM20bis should remain within 5% on representative radii."""
    m = make_model(n)
    r_arr = np.logspace(-2, 1, 15) * RE_PC
    rho_vm20bis = m.density_3d_VM20bis(r_arr)
    rho_num = m.density_3d_numerical(r_arr)
    valid = (rho_num > 0) & (rho_vm20bis > 0)
    rel_err = np.abs(rho_vm20bis[valid] / rho_num[valid] - 1.0)
    assert rel_err.size > 0
    assert np.all(rel_err < 0.05), f"n={n}: max rel err = {rel_err.max():.2%}"


def test_vm20bis_extended_low_radius():
    m = make_model(1.0)
    rho = m.density_3d_VM20bis(RE_PC * 1e-4)
    assert np.isfinite(rho) and rho > 0


def test_vm20bis_out_of_range_n():
    m = SersicModel(re_pc=RE_PC, n=4.0)
    with pytest.raises(ValueError, match="3.4"):
        m.density_3d_VM20bis(RE_PC)


def test_vm20bis_out_of_range_r():
    m = make_model(2.0)
    with pytest.raises(ValueError, match="r/R_e"):
        m.density_3d_VM20bis(RE_PC * 1e-5)


def test_density_3d_lgm_out_of_range():
    m_low = SersicModel(re_pc=RE_PC, n=0.3)
    m_high = SersicModel(re_pc=RE_PC, n=12.0)
    with pytest.raises(ValueError):
        m_low.density_3d_LGM(RE_PC)
    with pytest.raises(ValueError):
        m_high.density_3d_LGM(RE_PC)


# ---------------------------------------------------------------------------
# Public dispatch and safe auto policy
# ---------------------------------------------------------------------------


def test_density_3d_default_is_auto():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    assert m.deprojection_method == "auto"
    assert m.density_3d(RE_PC) == pytest.approx(m.density_3d_VM20bis(RE_PC), rel=1e-12)


def test_auto_low_n_uses_official_vm20bis_inside_domain():
    m = SersicModel(re_pc=RE_PC, n=0.75)
    r = 0.03 * RE_PC
    assert m.density_3d(r, method="auto") == pytest.approx(
        m.density_3d_VM20bis(r), rel=1e-12
    )


def test_auto_high_n_uses_sp04_inside_domain():
    m = SersicModel(re_pc=RE_PC, n=4.0)
    r = RE_PC
    expected = sp04_density(r, re_pc=RE_PC, n=4.0, b=m.b)
    assert m.density_3d(r, method="auto") == pytest.approx(expected, rel=1e-12)


def test_auto_falls_back_to_numerical_below_calibrated_radius():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    r = RE_PC * 1e-5
    assert m.density_3d(r, method="auto") == pytest.approx(
        m.density_3d_numerical(r), rel=1e-12
    )


def test_auto_zero_uses_numerical_center_limit():
    m = SersicModel(re_pc=RE_PC, n=0.75)
    assert m.density_3d(0.0, method="auto") == pytest.approx(
        m.density_3d_numerical(0.0), rel=1e-12
    )


def test_auto_mixed_array_dispatch_preserves_shape():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    r = np.array([RE_PC * 1e-5, RE_PC, np.inf])
    result = m.density_3d(r, method="auto")
    assert result.shape == r.shape
    assert result[0] == pytest.approx(m.density_3d_numerical(r[0]), rel=1e-12)
    assert result[1] == pytest.approx(m.density_3d_VM20bis(r[1]), rel=1e-12)
    assert result[2] == 0.0


def test_auto_negative_raises():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    with pytest.raises(ValueError, match="non-negative"):
        m.density_3d(-1.0, method="auto")


def test_density_3d_method_approx():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    assert m.density_3d(RE_PC, method="approx") == pytest.approx(
        m.density_3d_LGM(RE_PC), rel=1e-12
    )


def test_density_3d_method_vm20():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    assert m.density_3d(RE_PC, method="vm20") == pytest.approx(
        m.density_3d_VM20(RE_PC), rel=1e-12
    )


def test_density_3d_method_vm20bis():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    assert m.density_3d(RE_PC, method="vm20bis") == pytest.approx(
        m.density_3d_VM20bis(RE_PC), rel=1e-12
    )


def test_density_3d_method_numerical():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    assert m.density_3d(RE_PC, method="numerical") == pytest.approx(
        m.density_3d_numerical(RE_PC), rel=1e-10
    )


def test_instance_deprojection_method_numerical():
    m = SersicModel(re_pc=RE_PC, n=2.0, deprojection_method="numerical")
    assert m.density_3d(RE_PC) == pytest.approx(m.density_3d_numerical(RE_PC), rel=1e-10)


def test_instance_deprojection_method_vm20bis():
    m = SersicModel(re_pc=RE_PC, n=2.0, deprojection_method="vm20bis")
    assert m.density_3d(RE_PC) == pytest.approx(m.density_3d_VM20bis(RE_PC), rel=1e-12)


def test_invalid_method_raises():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    with pytest.raises(ValueError, match="method"):
        m.density_3d(RE_PC, method="bad_method")


def test_invalid_deprojection_method_at_construction():
    with pytest.raises(ValueError):
        SersicModel(re_pc=RE_PC, n=2.0, deprojection_method="invalid")


def test_density_3d_numerical_scalar():
    result = make_model(2.0).density_3d_numerical(RE_PC)
    assert np.isscalar(result) or np.ndim(result) == 0
    assert result > 0


def test_density_3d_numerical_array():
    m = make_model(2.0)
    r = np.array([50.0, 100.0, 200.0])
    result = m.density_3d_numerical(r)
    assert result.shape == r.shape
    assert np.all(result > 0)


def test_density_3d_vm20_scalar():
    result = SersicModel(re_pc=RE_PC, n=2.0).density_3d_VM20(RE_PC)
    assert np.isscalar(result) or np.ndim(result) == 0
    assert result > 0


def test_density_3d_vm20_array():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    r = np.array([50.0, 100.0, 200.0])
    result = m.density_3d_VM20(r)
    assert result.shape == r.shape
    assert np.all(result > 0)


def test_public_sersic_class_remains_in_jeanspy_model_namespace():
    """The implementation split must not change the public import path."""
    assert SersicModel.__module__ == "jeanspy.model"
