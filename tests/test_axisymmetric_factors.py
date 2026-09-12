"""Finite-cone geometry and cusp-integrability checks for J and D factors."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from jeanspy.axisymmetric import ZhaoHalo
from jeanspy.axisymmetric_factors import C_J, C_D
from jeanspy.model import NFWModel


@pytest.mark.parametrize("roi_deg", [.1, .5, 5.])
@pytest.mark.parametrize("inclination", [0., .8, np.pi/2])
def test_spherical_ullio_limit(roi_deg, inclination):
    halo = ZhaoHalo(.1, 500., r_t_pc=1000.)
    spherical = NFWModel(rhos_Msunpc3=.1, rs_pc=500., r_t_pc=1000.)
    expected = spherical.jfactor_ullio2016(80000., roi_deg)
    value = halo.jfactor(80000., roi_deg, inclination=inclination)
    np.testing.assert_allclose(value, expected, rtol=7e-4)


def _ray_reference(halo, dist, roi, inclination, power):
    """Independent observer-ray integral, using ellipsoid intersections.

    Production integrates spheroidal volume shells instead. This reference
    uses adaptive LOS and polar quadrature plus periodic sky azimuth nodes.
    Test apertures lie entirely within the projected halo to avoid a grazing
    boundary in the reference integration.
    """
    si, ci = np.sin(inclination), np.cos(inclination)
    theta_max = np.deg2rad(roi)
    def one_azimuth(phi):
        e0 = np.array([0., si, ci/halo.Q])
        e1 = np.array([np.cos(phi), np.sin(phi)*ci, -np.sin(phi)*si/halo.Q])
        cross2 = np.dot(np.cross(e0, e1), np.cross(e0, e1))
        def one_radius(u):
            theta = theta_max*u*u
            ray = np.cos(theta)*e0+np.sin(theta)*e1
            A = np.dot(ray, ray)
            closest = dist*np.sin(theta)*np.sqrt(cross2/A)
            # m = closest*cosh(t), dLOS = closest*cosh(t)/sqrt(A) dt
            upper = np.arccosh(halo.r_t_pc/closest)
            def integrand(t):
                m = closest*np.cosh(t)
                return halo._density_slope(m)[0]**power*m/np.sqrt(A)
            los = 2*quad(integrand, 0., upper, epsabs=0., epsrel=1e-8)[0]
            return np.sin(theta)*los*2*theta_max*u
        return quad(one_radius, 0., 1., epsabs=0., epsrel=1e-7)[0]
    return 2*np.pi/32*sum(one_azimuth(phi) for phi in 2*np.pi*(np.arange(32)+.5)/32)


@pytest.mark.parametrize("Q,inc", [(.55, .9), (1.4, 1.2)])
@pytest.mark.parametrize("power,conversion,method", [(2, C_J, "jfactor"), (1, C_D, "dfactor")])
def test_flattened_factors_against_independent_observer_rays(Q, inc, power, conversion, method):
    halo = ZhaoHalo(.05, 500., Q=Q, alpha=1.2, beta=3.5, gamma=.8, r_t_pc=3000.)
    expected = conversion*_ray_reference(halo, 30000., .35, inc, power)
    value = getattr(halo, method)(30000., .35, inclination=inc, n_mu=160, n_phi=160)
    np.testing.assert_allclose(value, expected, rtol=1.5e-3)


def test_full_halo_distant_limit_and_exact_distance_correction():
    halo = ZhaoHalo(.1, 500., Q=.6, alpha=2., beta=5., gamma=0., r_t_pc=1000.)
    dist = 1e7
    expected_d = C_D*halo.enclosed_mass(np.inf)/dist**2
    np.testing.assert_allclose(halo.dfactor(dist, 1.), expected_d, rtol=1e-8)
    volume = 4*np.pi*halo.Q*quad(lambda m: m*m*halo._density_slope(m)[0]**2,
                                0., halo.r_t_pc, epsabs=0., epsrel=1e-10)[0]
    np.testing.assert_allclose(halo.jfactor(dist, 1.), C_J*volume/dist**2, rtol=1e-8)
    nearby = replace(halo, Q=1.)
    reference = quad(lambda r: 4*np.pi*r/2000*np.arctanh(r/2000)*nearby._density_slope(r)[0],
                     0., nearby.r_t_pc, epsabs=0., epsrel=1e-10)[0]
    np.testing.assert_allclose(nearby.dfactor(2000., 60.), C_D*reference, rtol=1e-10)


def test_slope_integrability_and_density_scaling():
    halo = ZhaoHalo(.1, 500., Q=.7, gamma=1.49, r_t_pc=1000.)
    j = halo.jfactor(80000., 2.)
    assert np.isfinite(j) and j > 0
    np.testing.assert_allclose(replace(halo, rho_s=.2).jfactor(80000., 2.), 4*j)
    np.testing.assert_allclose(replace(halo, rho_s=.2).dfactor(80000., 2.), 2*halo.dfactor(80000., 2.))
    with pytest.raises(ValueError, match="diverges"):
        replace(halo, gamma=1.5).jfactor(80000., .5)
    assert np.isfinite(replace(halo, gamma=1.9).dfactor(80000., .5))


def test_factor_domains_and_zero_aperture():
    with pytest.raises(ValueError, match="finite ellipsoidal"):
        ZhaoHalo(.1, 500.).jfactor(80000., .5)
    halo = ZhaoHalo(.1, 500., Q=1.5, r_t_pc=1000.)
    for dist, roi in [(1000., .5), (80000., -1.), (80000., 90.), (np.inf, .5)]:
        with pytest.raises(ValueError):
            halo.jfactor(dist, roi)
    assert halo.jfactor(80000., 0.) == 0.


def test_aperture_crossing_cutoff_and_angular_refinement():
    halo = ZhaoHalo(.1, 500., Q=.55, gamma=.8, r_t_pc=3000.)
    for method in (halo.jfactor, halo.dfactor):
        coarse = method(30000., 4.8, inclination=.9, n_mu=64, n_phi=64)
        fine = method(30000., 4.8, inclination=.9, n_mu=192, n_phi=192)
        np.testing.assert_allclose(coarse, fine, rtol=2e-3)
        # Once the full halo is inside the cone, increasing its angle adds no mass.
        np.testing.assert_allclose(method(30000., 6., inclination=.9),
                                   method(30000., 10., inclination=.9), rtol=1e-12)
