"""Regression checks for the published Vitral & Mamon (2021) VM20bis table."""

from importlib.resources import files

import numpy as np


def test_vm20bis_uses_published_density_coefficients():
    """Pin representative values from the authors' public ``coeff_dens.txt``.

    Source: https://gitlab.com/eduardo-vitral/vitral_mamon_2020b
    The table is bundled as ``jeanspy/data/coeff_dens_vm20bis.csv``.
    """
    path = files("jeanspy").joinpath("data", "coeff_dens_vm20bis.csv")
    coeff = np.loadtxt(path, comments="#", delimiter=None)

    assert coeff.shape == (11, 11)
    np.testing.assert_allclose(
        coeff[0],
        [
            5.059e-03,
            -1.163e-03,
            -8.712e-02,
            1.743e-01,
            1.193e00,
            -1.287e00,
            -1.032e01,
            4.356e00,
            4.310e01,
            -2.048e01,
            -3.831e01,
        ],
        rtol=0.0,
        atol=5e-15,
    )
    assert coeff[1, 9] == -47.67
    assert coeff[7, 3] == 1.825e-4
    assert coeff[10, 0] == -3.430e-7
