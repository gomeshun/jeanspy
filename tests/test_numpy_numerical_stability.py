"""Independent references for small-radius mass and low-index Sersic tails."""

import warnings

import numpy as np
import pytest
from scipy.constants import parsec
from scipy.integrate import IntegrationWarning, quad

from jeanspy.model import (
    ConstantAnisotropyModel, DSphModel, GMsun_m3s2, NFWModel, SersicModel,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nfw_small_radius_mass_matches_positive_integral(dtype):
    halo = NFWModel(rs_pc=1000., rhos_Msunpc3=.01, r_t_pc=10000.)
    radius = (1000*np.array([0., 1e-8, 1e-7, 1.19263525e-7, 1e-6,
                            1e-5, 1e-4, 1e-3, 1e-2, .0999, .1, .1001,
                            1., 10.])).astype(dtype)
    # M/(4*pi*rho_s*r_s^3) = x^2*integral_0^1 t/(1+x*t)^2 dt.
    # This independent integral never subtracts nearly equal terms.
    x = radius.astype(np.float64)/1000.
    expected = np.array([v*v*quad(lambda t: t/(1+v*t)**2, 0., 1.,
                                 epsabs=0., epsrel=1e-12)[0] for v in x])
    expected *= 4*np.pi*.01*1000.**3
    actual = halo.enclosed_mass(radius)
    np.testing.assert_allclose(actual, expected,
                               rtol=3e-6 if dtype == np.float32 else 3e-10,
                               atol=0.)
    assert actual[0] == 0.
    assert np.all(actual[1:] > 0.)


def test_float32_nfw_mass_is_increasing_near_old_series_boundary():
    halo = NFWModel(rs_pc=1000., rhos_Msunpc3=.01, r_t_pc=10000.)
    radii = np.geomspace(.0001, .1, 10000).astype(np.float32)
    assert np.all(np.diff(halo.enclosed_mass(radii)) > 0.)


@pytest.mark.parametrize("index", [.2, .3, .49])
def test_low_index_sersic_far_tail_is_zero(index):
    tracer = SersicModel(re_pc=200., n=index)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", IntegrationWarning)
        np.testing.assert_array_equal(tracer.density_3d([1e180, 1e250, 1e300]), 0.)


def _independent_low_index_sersic_los(tracer, halo, projected_radius, upper):
    """Abel deproject in z, then project the isotropic Jeans kernel in z.

    Neither integration uses the package's theta Abel inversion or DE nodes.
    The finite 5/10-Re bounds have negligible exponentially suppressed tails.
    """
    re, index = tracer.params.re_pc, tracer.params.n
    b, surface_norm = float(tracer.b), float(tracer.norm)

    def density(radius):
        def integrand(t):
            x = np.hypot(radius/re, t)
            return np.exp(-b*x**(1/index))*x**(1/index-2)
        integral = quad(integrand, 0., 10., epsabs=0., epsrel=1e-10)[0]
        return b/(index*np.pi*re*surface_norm)*integral

    def integrand(t):
        x = np.hypot(projected_radius/re, t)
        radius = re*x
        return density(radius)*float(halo.enclosed_mass(radius))*t*t/x**3

    integral = quad(integrand, 0., upper, epsabs=0., epsrel=1e-9)[0]
    return (2*GMsun_m3s2*1e-6/parsec
            /float(tracer.density_2d(projected_radius))*integral)


def test_default_low_index_sersic_los_matches_independent_integral():
    tracer = SersicModel(re_pc=200., n=.3)
    halo = NFWModel(rs_pc=1000., rhos_Msunpc3=.01, r_t_pc=10000.)
    model = DSphModel(vmem_kms=0., submodels={
        "StellarModel": tracer, "DMModel": halo,
        "AnisotropyModel": ConstantAnisotropyModel(beta_ani=0.),
    })
    expected = _independent_low_index_sersic_los(tracer, halo, 100., 10.)
    refined = _independent_low_index_sersic_los(tracer, halo, 100., 5.)
    assert expected == pytest.approx(refined, rel=1e-10)
    assert expected > 0.
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        actual = model.sigmalos2(100.)
    assert actual == pytest.approx(expected, rel=1e-5)


def test_numpy_los_promotes_float32_radii_before_evaluating_halo_mass():
    seen_dtypes = []

    class RecordingNFW(NFWModel):
        def enclosed_mass(self, radius):
            seen_dtypes.append(np.asarray(radius).dtype)
            return super().enclosed_mass(radius)

    from jeanspy.model import PlummerModel
    model = DSphModel(vmem_kms=0., submodels={
        "StellarModel": PlummerModel(re_pc=200.),
        "DMModel": RecordingNFW(rs_pc=1000., rhos_Msunpc3=.01, r_t_pc=10000.),
        "AnisotropyModel": ConstantAnisotropyModel(beta_ani=0.),
    })
    assert np.isfinite(model.sigmalos2(np.array([100.], dtype=np.float32))).all()
    assert seen_dtypes and all(dtype == np.float64 for dtype in seen_dtypes)
