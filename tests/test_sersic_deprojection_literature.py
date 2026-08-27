"""Independent literature checks for Sérsic deprojection benchmarks."""

import numpy as np
import pytest

from jeanspy._sersic_deprojection import (
    ciotti2025_density,
    ciotti2025_matching_p,
    sp04_density,
)
from jeanspy.model import SersicModel


RE_PC = 100.0


def _model(n):
    return SersicModel(re_pc=RE_PC, n=n, deprojection_method="numerical")


@pytest.mark.parametrize(
    ("n", "published_p"),
    [
        (0.55, 13.19),
        (0.75, 2.383),
        (1.50, 1.511),
        (2.00, 1.718),
        (4.00, 1.955),
        (8.00, 2.052),
    ],
)
def test_ciotti2025_matching_p_reproduces_table_b1(n, published_p):
    """Luminosity-conservation root reproduces Ciotti+2025 Table B.1."""
    m = _model(n)
    p = ciotti2025_matching_p(n, m.b)
    assert p == pytest.approx(published_p, abs=5e-3)


@pytest.mark.parametrize("n", [0.5, 1.0])
def test_ciotti2025_special_exact_cases_match_abel(n):
    """The Gaussian (n=0.5) and K0 (n=1) branches match Abel inversion."""
    m = _model(n)
    radii = np.logspace(-1.3, 0.6, 16) * RE_PC
    rho_lit = ciotti2025_density(radii, re_pc=RE_PC, n=n, b=m.b)
    rho_num = m.density_3d_numerical(radii)
    rel = np.abs(rho_lit / rho_num - 1.0)
    assert np.max(rel) < 2e-4


@pytest.mark.parametrize(
    ("n", "max_error"),
    [
        (2.0, 0.05),
        (4.0, 0.015),
        (8.0, 0.008),
    ],
)
def test_ciotti2025_accuracy_improves_at_high_n(n, max_error):
    """Published matched approximation is accurate and improves with n."""
    m = _model(n)
    x = np.logspace(-3, 2, 30)
    radii = x * RE_PC
    rho_lit = ciotti2025_density(radii, re_pc=RE_PC, n=n, b=m.b)
    rho_num = m.density_3d_numerical(radii)
    valid = (rho_num > 0) & np.isfinite(rho_num) & np.isfinite(rho_lit)
    rel = np.abs(rho_lit[valid] / rho_num[valid] - 1.0)
    assert rel.size > 0
    assert np.max(rel) < max_error


@pytest.mark.parametrize("n", [3.4, 4.0, 8.0])
def test_sp04_is_high_accuracy_in_vm21_high_n_hybrid_regime(n):
    """SP04 is accurate in the n>3.4 regime selected by Vitral & Mamon 2021."""
    m = _model(n)
    x = np.logspace(-3, 2, 30)
    radii = x * RE_PC
    rho_sp = sp04_density(radii, re_pc=RE_PC, n=n, b=m.b)
    rho_num = m.density_3d_numerical(radii)
    valid = (rho_num > 0) & np.isfinite(rho_num) & np.isfinite(rho_sp)
    rel = np.abs(rho_sp[valid] / rho_num[valid] - 1.0)
    assert rel.size > 0
    assert np.max(rel) < 0.02


def test_sp04_rejects_n_at_or_below_one():
    m = _model(1.0)
    with pytest.raises(ValueError, match="n > 1"):
        sp04_density(RE_PC, re_pc=RE_PC, n=1.0, b=m.b)
