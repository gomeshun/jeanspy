"""Mass/force/Jeans consistency at a finite spheroidal halo boundary."""
import numpy as np
import pytest

from jeanspy.axisymmetric import AxisymmetricDSphModel, ZhaoHalo, G


@pytest.mark.parametrize("Q", [.45, 1., 1.7])
def test_truncated_force_derivative_and_poisson(Q):
    halo = ZhaoHalo(.1, 500., Q=Q, r_t_pc=600.)
    for R, z in [(100., 40.), (850., 900.)]:
        h = .01
        gr, gz, mixed = halo._gradients(R, z, 128)
        radial = (halo.potential_gradient(R+h, z, 128)[0]-halo.potential_gradient(R-h, z, 128)[0])/(2*h)
        vertical = (halo.potential_gradient(R, z+h, 128)[1]-halo.potential_gradient(R, z-h, 128)[1])/(2*h)
        mixed_fd = (halo.potential_gradient(R+h, z, 128)[1]-halo.potential_gradient(R-h, z, 128)[1])/(2*h)
        np.testing.assert_allclose(mixed, mixed_fd, rtol=2e-7, atol=1e-12)
        np.testing.assert_allclose(radial+gr/R+vertical, 4*np.pi*G*halo.density(R, z), rtol=2e-7, atol=1e-11)


def test_spherical_force_mass_and_boundary_continuity():
    halo = ZhaoHalo(.1, 500., r_t_pc=600.)
    r = np.array([1., 100., 600., 700., 10000.])
    x = np.minimum(r, 600.)/500.
    mass = 4*np.pi*.1*500**3*(np.log1p(x)-x/(1+x))
    np.testing.assert_allclose(halo.enclosed_mass(r), mass, rtol=1e-9)
    np.testing.assert_allclose(halo.potential_gradient(r, 0)[0], G*mass/r**2, rtol=1e-9)
    np.testing.assert_allclose(halo.enclosed_mass(np.inf), mass[-1], rtol=1e-12)
    assert halo.density(601., 0.) == 0.
    around = halo.potential_gradient(np.array([600*(1-1e-8), 600*(1+1e-8)]), 0)[0]
    np.testing.assert_allclose(around[0], around[1], rtol=4e-8)


def test_truncated_spherical_jeans_limit():
    from jeanspy.model import DSphModel, PlummerModel, NFWModel, ConstantAnisotropyModel
    spherical = DSphModel(submodels=dict(StellarModel=PlummerModel(), DMModel=NFWModel(),
                                         AnisotropyModel=ConstantAnisotropyModel()))
    physical = dict(re_pc=300., rs_pc=500., rhos_Msunpc3=.1, r_t_pc=600., beta_ani=0., vmem_kms=0.)
    spherical.update(physical)
    p = {k: v for k, v in physical.items() if k != "beta_ani"}
    p.update(q=1., Q=1., beta_z=0., inclination=.8)
    axis = AxisymmetricDSphModel(96, 160, 160)
    radius = np.array([50., 300., 900.])
    np.testing.assert_allclose(axis.sigmalos2(radius, 0., params=p), spherical.sigmalos2(radius), rtol=6e-4)


def test_mass_scales_with_ellipsoid_volume():
    spherical = ZhaoHalo(.1, 500., r_t_pc=600.)
    oblate = ZhaoHalo(.1, 500., Q=.6, r_t_pc=600.)
    np.testing.assert_allclose(oblate.enclosed_mass([10., 600., 800.]),
                               .6*spherical.enclosed_mass([10., 600., 800.]))
