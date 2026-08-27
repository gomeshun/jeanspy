"""Tests for SersicModel.density_3d, density_3d_LGM, and density_3d_numerical."""
import numpy as np
import pytest
from scipy.special import gamma, k0

from jeanspy.model import SersicModel


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

RE_PC = 100.0  # arbitrary effective radius in parsec


def make_model(n, deprojection_method="approx"):
    return SersicModel(re_pc=RE_PC, n=n, deprojection_method=deprojection_method)


def _radii(rmin=0.01, rmax=10.0, num=20):
    """Log-spaced r/R_e values, returned as r in parsec."""
    return np.logspace(np.log10(rmin), np.log10(rmax), num) * RE_PC


# ---------------------------------------------------------------------------
# 1. Exact n=0.5 (Gaussian projected → Gaussian 3D)
# ---------------------------------------------------------------------------

def exact_rho_n05(r_pc, re_pc):
    """Exact 3-D deprojection of the n=0.5 Sérsic profile (= Gaussian).

    For n=0.5, Sigma(R) = exp(-b*(R/re)) / norm2d with b = b(n=0.5).
    The Abel inversion of a Gaussian surface density Sigma(R) = A*exp(-R^2/(2*sigma^2))
    gives rho(r) = A / (sqrt(2*pi)*sigma) * exp(-r^2/(2*sigma^2)).

    For the Sérsic n=0.5 profile: Sigma(R) = (1/norm) * exp(-b*(R/re)^2),
    i.e. a Gaussian with sigma^2 = re^2/(2b).
    """
    m = make_model(0.5)
    b = m.b
    norm2d = m.norm
    # Sigma(R) = exp(-b*(R/re)^2) / norm2d  =>  sigma^2 = re^2 / (2b)
    sigma2 = re_pc**2 / (2 * b)
    A = 1.0 / norm2d
    return A / (np.sqrt(2 * np.pi * sigma2)) * np.exp(-r_pc**2 / (2 * sigma2))


def test_numerical_n05_vs_exact():
    m = make_model(0.5)
    r_arr = _radii(0.05, 3.0, 20)
    rho_num = m.density_3d_numerical(r_arr)
    rho_exact = exact_rho_n05(r_arr, RE_PC)
    # Avoid underflow region
    valid = rho_exact > rho_exact.max() * 1e-8
    rel_err = np.abs(rho_num[valid] / rho_exact[valid] - 1.0)
    assert np.all(rel_err < 1e-4), f"max rel err = {rel_err.max():.2e}"


# ---------------------------------------------------------------------------
# 2. Exact n=1 (exponential projected → K0 3D)
# ---------------------------------------------------------------------------

def exact_rho_n1(r_pc, re_pc):
    """Exact 3-D deprojection of the n=1 Sérsic profile (exponential disc).

    For n=1, Sigma(R) = exp(-b*(R/re)) / norm2d with b = b(n=1).
    The Abel inversion gives rho(r) = (b/re) / (pi*norm2d) * K0(b*r/re),
    where K0 is the modified Bessel function of the second kind.
    """
    m = make_model(1.0)
    b = m.b
    norm2d = m.norm
    return (b / re_pc) / (np.pi * norm2d) * k0(b * r_pc / re_pc)


def test_numerical_n1_vs_exact():
    m = make_model(1.0)
    r_arr = _radii(0.05, 5.0, 25)
    rho_num = m.density_3d_numerical(r_arr)
    rho_exact = exact_rho_n1(r_arr, RE_PC)
    valid = rho_exact > rho_exact.max() * 1e-8
    rel_err = np.abs(rho_num[valid] / rho_exact[valid] - 1.0)
    assert np.all(rel_err < 1e-4), f"max rel err = {rel_err.max():.2e}"


# ---------------------------------------------------------------------------
# 3. Approx vs numerical around r ~ R_e
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1.0, 2.0, 4.0, 8.0])
def test_approx_vs_numerical_near_re(n):
    """Near r ~ R_e the LGM approximation and numerical result agree at ~percent level."""
    m = make_model(n)
    # Use a few radii near R_e
    r_arr = np.array([0.5, 1.0, 2.0]) * RE_PC
    rho_lgm = m.density_3d_LGM(r_arr)
    rho_num = m.density_3d_numerical(r_arr)
    rel_err = np.abs(rho_lgm / rho_num - 1.0)
    assert np.all(rel_err < 0.05), (
        f"n={n}: max rel err near R_e = {rel_err.max():.2%}"
    )


# Low-n (n=0.6): agreement only near R_e, not at small radii
def test_approx_vs_numerical_low_n_near_re():
    n = 0.6
    m = make_model(n)
    r_arr = np.array([0.5, 1.0, 2.0]) * RE_PC
    rho_lgm = m.density_3d_LGM(r_arr)
    rho_num = m.density_3d_numerical(r_arr)
    rel_err = np.abs(rho_lgm / rho_num - 1.0)
    # Near R_e a percent-level agreement is expected even for low n
    assert np.all(rel_err < 0.10), (
        f"n={n}: max rel err near R_e = {rel_err.max():.2%}"
    )


# ---------------------------------------------------------------------------
# 4. API behavior
# ---------------------------------------------------------------------------

def test_density_3d_default_matches_lgm():
    """density_3d() with default settings should match density_3d_LGM()."""
    m = make_model(2.0)
    r = 100.0
    assert m.density_3d(r) == pytest.approx(m.density_3d_LGM(r), rel=1e-12)


def test_density_3d_method_numerical():
    """density_3d(method='numerical') should match density_3d_numerical()."""
    m = make_model(2.0)
    r = 100.0
    assert m.density_3d(r, method="numerical") == pytest.approx(
        m.density_3d_numerical(r), rel=1e-10
    )


def test_instance_deprojection_method():
    """deprojection_method='numerical' at construction selects numerical by default."""
    m = make_model(2.0, deprojection_method="numerical")
    r = 100.0
    assert m.density_3d(r) == pytest.approx(m.density_3d_numerical(r), rel=1e-10)


def test_invalid_method_raises():
    m = make_model(2.0)
    with pytest.raises(ValueError, match="method"):
        m.density_3d(100.0, method="bad_method")


def test_invalid_deprojection_method_at_construction():
    with pytest.raises(ValueError):
        SersicModel(re_pc=RE_PC, n=2.0, deprojection_method="invalid")


def test_density_3d_numerical_scalar():
    m = make_model(2.0)
    result = m.density_3d_numerical(RE_PC)
    assert np.isscalar(result) or np.ndim(result) == 0
    assert result > 0


def test_density_3d_numerical_array():
    m = make_model(2.0)
    r = np.array([50.0, 100.0, 200.0])
    result = m.density_3d_numerical(r)
    assert result.shape == r.shape
    assert np.all(result > 0)


def test_density_3d_numerical_negative_raises():
    m = make_model(2.0)
    with pytest.raises(ValueError, match="non-negative"):
        m.density_3d_numerical(-1.0)


def test_density_3d_lgm_out_of_range():
    """density_3d_LGM should raise ValueError outside n ∈ [0.5, 10]."""
    m_low = SersicModel(re_pc=RE_PC, n=0.3)
    m_high = SersicModel(re_pc=RE_PC, n=12.0)
    with pytest.raises(ValueError):
        m_low.density_3d_LGM(RE_PC)
    with pytest.raises(ValueError):
        m_high.density_3d_LGM(RE_PC)
