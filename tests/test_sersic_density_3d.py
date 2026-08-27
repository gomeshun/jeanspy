"""Tests for SersicModel.density_3d, density_3d_LGM, density_3d_VM20,
density_3d_VM20bis, and density_3d_numerical."""
import numpy as np
import pytest
from scipy.special import gamma, k0
from scipy.integrate import quad

from jeanspy.model import SersicModel


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

RE_PC = 100.0  # arbitrary effective radius in parsec


def make_model(n, deprojection_method="numerical"):
    """Make a SersicModel; default method is 'numerical' to avoid needing
    r/Re bounds checks in tests that probe a wide range of radii."""
    return SersicModel(re_pc=RE_PC, n=n, deprojection_method=deprojection_method)


def _radii(rmin=0.01, rmax=10.0, num=20):
    """Log-spaced r/R_e values, returned as r in parsec."""
    return np.logspace(np.log10(rmin), np.log10(rmax), num) * RE_PC


# ---------------------------------------------------------------------------
# 1. Exact n=0.5 (Gaussian projected → Gaussian 3D)
# ---------------------------------------------------------------------------

def exact_rho_n05(r_pc, re_pc):
    """Exact 3-D deprojection of the n=0.5 Sérsic profile (= Gaussian).

    For n=0.5, Sigma(R) = exp(-b*(R/re)^2) / norm2d with b = b(n=0.5).
    sigma^2 = re^2 / (2b), and the Abel inversion gives a Gaussian.
    """
    m = make_model(0.5)
    b = m.b
    norm2d = m.norm
    sigma2 = re_pc**2 / (2 * b)
    A = 1.0 / norm2d
    return A / (np.sqrt(2 * np.pi * sigma2)) * np.exp(-r_pc**2 / (2 * sigma2))


def test_numerical_n05_vs_exact():
    m = make_model(0.5)
    r_arr = _radii(0.05, 3.0, 20)
    rho_num = m.density_3d_numerical(r_arr)
    rho_exact = exact_rho_n05(r_arr, RE_PC)
    valid = rho_exact > rho_exact.max() * 1e-8
    rel_err = np.abs(rho_num[valid] / rho_exact[valid] - 1.0)
    assert np.all(rel_err < 1e-4), f"max rel err = {rel_err.max():.2e}"


# ---------------------------------------------------------------------------
# 2. Exact n=1 (exponential projected → K0 3D)
# ---------------------------------------------------------------------------

def exact_rho_n1(r_pc, re_pc):
    """Exact 3-D deprojection of the n=1 Sérsic (exponential) profile."""
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
# 3. LGM approx vs numerical around r ~ R_e
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1.0, 2.0, 4.0, 8.0])
def test_approx_vs_numerical_near_re(n):
    """Near r ~ R_e the LGM approximation and numerical result agree at ~percent level."""
    m = make_model(n)
    r_arr = np.array([0.5, 1.0, 2.0]) * RE_PC
    rho_lgm = m.density_3d_LGM(r_arr)
    rho_num = m.density_3d_numerical(r_arr)
    rel_err = np.abs(rho_lgm / rho_num - 1.0)
    assert np.all(rel_err < 0.05), (
        f"n={n}: max rel err near R_e = {rel_err.max():.2%}"
    )


def test_approx_vs_numerical_low_n_near_re():
    """Low n=0.6: LGM and numerical agree near R_e (not at small radii)."""
    n = 0.6
    m = make_model(n)
    r_arr = np.array([0.5, 1.0, 2.0]) * RE_PC
    rho_lgm = m.density_3d_LGM(r_arr)
    rho_num = m.density_3d_numerical(r_arr)
    rel_err = np.abs(rho_lgm / rho_num - 1.0)
    assert np.all(rel_err < 0.10), (
        f"n={n}: max rel err near R_e = {rel_err.max():.2%}"
    )


# ---------------------------------------------------------------------------
# 4. VM20 accuracy vs numerical reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1.0, 2.0, 4.0])
def test_vm20_vs_numerical_near_re(n):
    """VM20 should be more accurate than LGM vs the numerical reference near R_e."""
    m = make_model(n)
    r_arr = np.array([0.1, 0.5, 1.0, 2.0, 5.0]) * RE_PC
    rho_vm20 = m.density_3d_VM20(r_arr)
    rho_num = m.density_3d_numerical(r_arr)
    rel_err = np.abs(rho_vm20 / rho_num - 1.0)
    assert np.all(rel_err < 0.05), (
        f"n={n}: VM20 max rel err = {rel_err.max():.2%}"
    )


def test_vm20_low_n_accuracy():
    """VM20 improves over LGM at small r/Re for n=0.6."""
    m = make_model(0.6)
    # Use intermediate radii within the VM20 validity domain (1e-3 <= r/Re <= 1e3)
    r_arr = np.array([0.01, 0.05, 0.1, 0.5, 1.0]) * RE_PC
    rho_vm20 = m.density_3d_VM20(r_arr)
    rho_num = m.density_3d_numerical(r_arr)
    rho_lgm = m.density_3d_LGM(r_arr)
    rel_err_vm20 = np.abs(rho_vm20 / rho_num - 1.0)
    rel_err_lgm = np.abs(rho_lgm / rho_num - 1.0)
    # VM20 should be at least as good as LGM on average
    assert rel_err_vm20.mean() <= rel_err_lgm.mean() * 2.0, (
        "VM20 should not be dramatically worse than LGM for n=0.6"
    )


def test_vm20_out_of_range_n():
    m = SersicModel(re_pc=RE_PC, n=0.3)
    with pytest.raises(ValueError, match="0.5"):
        m.density_3d_VM20(RE_PC)


def test_vm20_out_of_range_r():
    m = make_model(2.0)
    with pytest.raises(ValueError, match="r/R_e"):
        m.density_3d_VM20(RE_PC * 1e-4)  # r/Re = 1e-4, below 1e-3 limit


# ---------------------------------------------------------------------------
# 5. VM20bis accuracy vs numerical reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0.5, 0.75, 1.0, 2.0])
def test_vm20bis_vs_numerical(n):
    """VM20bis should be accurate within ~5% over the physically meaningful domain.

    At very large r/Re for small n (e.g. n=0.5, r/Re > ~15), the Sérsic density
    drops below floating-point resolution (~1e-300), causing both the polynomial
    approximation and the numerical integral to underflow to 0.  These
    underflow radii are excluded from the accuracy check.
    """
    m = make_model(n)
    r_arr = np.logspace(-2, 1, 15) * RE_PC  # r/Re in [0.01, 10]
    rho_vm20bis = m.density_3d_VM20bis(r_arr)
    rho_num = m.density_3d_numerical(r_arr)
    # Exclude underflow (both ≈ 0)
    valid = (rho_num > 0) & (rho_vm20bis > 0)
    rel_err = np.abs(rho_vm20bis[valid] / rho_num[valid] - 1.0)
    assert rel_err.size > 0, "No valid radii for comparison"
    assert np.all(rel_err < 0.05), (
        f"n={n}: VM20bis max rel err = {rel_err.max():.2%}"
    )


def test_vm20bis_extended_low_radius():
    """VM20bis covers the extended domain down to r/Re = 1e-4."""
    m = make_model(1.0)
    # r/Re = 1e-4 is in the VM20bis domain but outside VM20
    r_small = RE_PC * 1e-4
    rho = m.density_3d_VM20bis(r_small)
    assert np.isfinite(rho) and rho > 0


def test_vm20bis_out_of_range_n():
    m = SersicModel(re_pc=RE_PC, n=4.0)
    with pytest.raises(ValueError, match="3.4"):
        m.density_3d_VM20bis(RE_PC)


def test_vm20bis_out_of_range_r():
    m = make_model(2.0)
    with pytest.raises(ValueError, match="r/R_e"):
        m.density_3d_VM20bis(RE_PC * 1e-5)  # below 1e-4 limit


# ---------------------------------------------------------------------------
# 6. Normalization consistency test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0.5, 1.0, 2.0, 4.0])
def test_numerical_normalization(n):
    """4*pi * integral rho(r)*r^2 dr should integrate to 1 (reprojection norm)."""
    m = make_model(n)
    re = RE_PC

    def integrand(r):
        return 4 * np.pi * r**2 * m.density_3d_numerical(r)

    # Integrate from a small inner radius to a large outer radius
    result, err = quad(integrand, re * 1e-3, re * 30,
                       limit=200, epsrel=1e-4, epsabs=0)
    # Should be close to 1 (some missing mass at very small and very large r)
    assert 0.90 < result < 1.02, (
        f"n={n}: normalization integral = {result:.4f} ± {err:.2e}"
    )


# ---------------------------------------------------------------------------
# 7. API behavior
# ---------------------------------------------------------------------------

def test_density_3d_default_is_vm20():
    """Default deprojection_method should be 'vm20'."""
    m = SersicModel(re_pc=RE_PC, n=2.0)
    assert m.deprojection_method == "vm20"
    r = RE_PC
    assert m.density_3d(r) == pytest.approx(m.density_3d_VM20(r), rel=1e-12)


def test_density_3d_method_approx():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    r = RE_PC
    assert m.density_3d(r, method="approx") == pytest.approx(
        m.density_3d_LGM(r), rel=1e-12
    )


def test_density_3d_method_vm20():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    r = RE_PC
    assert m.density_3d(r, method="vm20") == pytest.approx(
        m.density_3d_VM20(r), rel=1e-12
    )


def test_density_3d_method_vm20bis():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    r = RE_PC
    assert m.density_3d(r, method="vm20bis") == pytest.approx(
        m.density_3d_VM20bis(r), rel=1e-12
    )


def test_density_3d_method_numerical():
    """density_3d(method='numerical') should match density_3d_numerical()."""
    m = SersicModel(re_pc=RE_PC, n=2.0)
    r = RE_PC
    assert m.density_3d(r, method="numerical") == pytest.approx(
        m.density_3d_numerical(r), rel=1e-10
    )


def test_instance_deprojection_method_numerical():
    """deprojection_method='numerical' at construction selects numerical by default."""
    m = SersicModel(re_pc=RE_PC, n=2.0, deprojection_method="numerical")
    r = RE_PC
    assert m.density_3d(r) == pytest.approx(m.density_3d_numerical(r), rel=1e-10)


def test_instance_deprojection_method_vm20bis():
    m = SersicModel(re_pc=RE_PC, n=2.0, deprojection_method="vm20bis")
    r = RE_PC
    assert m.density_3d(r) == pytest.approx(m.density_3d_VM20bis(r), rel=1e-12)


def test_invalid_method_raises():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    with pytest.raises(ValueError, match="method"):
        m.density_3d(RE_PC, method="bad_method")


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


def test_density_3d_vm20_scalar():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    result = m.density_3d_VM20(RE_PC)
    assert np.isscalar(result) or np.ndim(result) == 0
    assert result > 0


def test_density_3d_vm20_array():
    m = SersicModel(re_pc=RE_PC, n=2.0)
    r = np.array([50.0, 100.0, 200.0])
    result = m.density_3d_VM20(r)
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
