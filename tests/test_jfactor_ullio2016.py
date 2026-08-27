import numpy as np
import pytest
from scipy.integrate import quad

from jeanspy.model import (
    C_J,
    DMModel,
    NFWModel,
    ZhaoModel,
    _ullio2016_inner_weight,
    _ullio2016_weight,
)


def _direct_line_of_sight_jfactor(model, dist_pc, roi_deg):
    r_t_pc = float(model.params.r_t_pc)
    b_max_pc = min(dist_pc * np.sin(np.deg2rad(roi_deg)), r_t_pc)

    def line_of_sight_integral(b_pc):
        z_max_pc = np.sqrt(r_t_pc**2 - b_pc**2)
        value, _ = quad(
            lambda z_pc: float(model.mass_density_3d(np.sqrt(b_pc**2 + z_pc**2))) ** 2,
            0.0,
            z_max_pc,
            epsabs=0.0,
            epsrel=2.0e-8,
            limit=300,
        )
        return 2.0 * value

    def impact_parameter_integrand(b_pc):
        return (
            2.0
            * np.pi
            * b_pc
            / (dist_pc * np.sqrt(dist_pc**2 - b_pc**2))
            * line_of_sight_integral(b_pc)
        )

    value, _ = quad(
        impact_parameter_integrand,
        0.0,
        b_max_pc,
        epsabs=0.0,
        epsrel=2.0e-7,
        limit=300,
    )
    return C_J * value


def test_full_ullio_matches_independent_line_of_sight_integral():
    model = ZhaoModel(
        rs_pc=500.0,
        rhos_Msunpc3=0.02,
        a=1.2,
        b=4.5,
        g=0.4,
        r_t_pc=8000.0,
    )
    dist_pc = 30000.0
    roi_deg = 2.0

    full = model.jfactor_ullio2016(dist_pc, roi_deg)
    reference = _direct_line_of_sight_jfactor(model, dist_pc, roi_deg)

    np.testing.assert_allclose(full, reference, rtol=3.0e-6)


def test_full_ullio_keeps_projected_outer_shells():
    model = ZhaoModel(
        rs_pc=500.0,
        rhos_Msunpc3=0.02,
        a=1.2,
        b=4.5,
        g=0.4,
        r_t_pc=8000.0,
    )

    full = model.jfactor_ullio2016(30000.0, 0.5)
    spherical = model.jfactor_ullio2016_simple(30000.0, 0.5)

    assert full > spherical


def test_simple_approximation_respects_physical_truncation():
    model = ZhaoModel(
        rs_pc=500.0,
        rhos_Msunpc3=0.02,
        a=1.2,
        b=4.5,
        g=0.4,
        r_t_pc=100.0,
    )
    dist_pc = 10000.0
    roi_deg = 1.0
    r_max_pc = min(dist_pc * np.sin(np.deg2rad(roi_deg)), model.params.r_t_pc)

    integral, _ = quad(
        lambda r_pc: r_pc**2 * float(model.mass_density_3d(r_pc)) ** 2,
        0.0,
        r_max_pc,
        epsabs=0.0,
        epsrel=1.0e-8,
        limit=300,
    )
    expected = C_J * 4.0 * np.pi / dist_pc**2 * integral

    np.testing.assert_allclose(
        model.jfactor_ullio2016_simple(dist_pc, roi_deg),
        expected,
        rtol=1.0e-7,
    )


def test_ullio_weight_endpoint_limits_are_stable():
    dist_pc = 1.0e9
    r_pc = 1.0e3
    expected_inner = (r_pc / dist_pc) * np.arctanh(r_pc / dist_pc)

    np.testing.assert_allclose(
        _ullio2016_inner_weight(r_pc, dist_pc),
        expected_inner,
        rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        _ullio2016_weight(r_pc, 0.0, np.nextafter(r_pc, 0.0), dist_pc),
        expected_inner,
        rtol=2.0e-8,
    )


def test_full_ullio_validates_geometry():
    model = NFWModel(rs_pc=1000.0, rhos_Msunpc3=0.01, r_t_pc=8000.0)

    invalid_geometry = [
        {"dist_pc": 0.0, "roi_deg": 0.5},
        {"dist_pc": -1.0, "roi_deg": 0.5},
        {"dist_pc": 30000.0, "roi_deg": 0.0},
        {"dist_pc": 30000.0, "roi_deg": -0.5},
        {"dist_pc": 30000.0, "roi_deg": np.nan},
        {"dist_pc": 8000.0, "roi_deg": 0.5},
        {"dist_pc": 30000.0, "roi_deg": 91.0},
    ]
    for kwargs in invalid_geometry:
        with pytest.raises(ValueError):
            model.jfactor_ullio2016(**kwargs)

    with pytest.raises(ValueError):
        model.jfactor_ullio2016_simple(30000.0, 1.1)

    assert np.isfinite(model.jfactor_ullio2016(30000.0, 2.0))


def test_jfactor_requires_a_finite_truncation_radius():
    class UntruncatedDM(DMModel):
        required_param_names = ["rho"]
        required_models = {}

        def mass_density_3d(self, r_pc):
            return self.params.rho

    model = UntruncatedDM(rho=1.0)

    with pytest.raises(ValueError, match="truncation radius"):
        model.jfactor_ullio2016(30000.0, 0.5)
    with pytest.raises(ValueError, match="truncation radius"):
        model.jfactor_ullio2016_simple(30000.0, 0.5)
