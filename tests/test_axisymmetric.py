"""Physical checks independent of the nested quadrature implementation."""
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from jeanspy.axisymmetric import AxisymmetricJeans, PlummerTracer, ZhaoHalo, G, intrinsic_axis_ratio


@pytest.fixture
def plummer():
    a, mass = 300., 1e7
    return AxisymmetricJeans(PlummerTracer(a), ZhaoHalo(
        3*mass/(4*np.pi*a**3), a, alpha=2, beta=5, gamma=0))


@pytest.mark.parametrize("inclination", [0., .7, np.pi/2])
def test_analytic_spherical_plummer(plummer, inclination):
    m = replace(plummer, inclination=inclination)
    R, z = np.array([0., 30., 300., 3000.]), np.array([0., 60., 0., 200.])
    expected = G*1e7/(6*np.sqrt(300**2 + R**2 + z**2))
    for moment in m.intrinsic_moments(R, z):
        assert_allclose(moment, expected, rtol=2e-7)
    x, y = np.array([0., 30., 300., 3000.]), 0.
    expected_los = 3*np.pi*G*1e7/(64*np.sqrt(300**2+x**2))
    assert_allclose(m.los_second_moment(x, y), expected_los, rtol=2e-6)


@pytest.mark.parametrize("q", [.35, .7, 1., 1.6])
@pytest.mark.parametrize("inc", [0., .6, np.pi/2])
def test_tracer_projection(q, inc):
    tracer = PlummerTracer(300., q)
    x, y = 120., -240.
    si, ci = np.sin(inc), np.cos(inc)
    value = quad(lambda l: tracer.density(np.hypot(x,y*ci+l*si),-y*si+l*ci),
                 -np.inf, np.inf, epsabs=1e-13, epsrel=1e-9)[0]
    assert_allclose(value, tracer.surface_density(x,y,inc), rtol=1e-8)


@pytest.mark.parametrize("Q", [.4, 1., 1.5])
def test_force_poisson_and_parity(Q):
    halo = ZhaoHalo(.1, 500., Q=Q)
    R, z, h = 240., 180., .02
    gR, gz = halo.potential_gradient(R,z)
    dR = (halo.potential_gradient(R+h,z)[0]-halo.potential_gradient(R-h,z)[0])/(2*h)
    dz = (halo.potential_gradient(R,z+h)[1]-halo.potential_gradient(R,z-h)[1])/(2*h)
    assert_allclose(dR+gR/R+dz, 4*np.pi*G*halo.density(R,z), rtol=2e-7)
    assert_allclose(halo.potential_gradient(R,-z), [gR,-gz])
    assert halo.potential_gradient(0,z)[0] == 0
    assert halo.potential_gradient(R,0)[1] == 0


def test_spherical_nfw_force():
    halo = ZhaoHalo(.1,500.)
    R, z = np.array([1.,100.,1000.]), np.array([2.,200.,2000.])
    r = np.hypot(R,z)
    x = r/500.
    mass = 4*np.pi*.1*500**3*(np.log1p(x)-x/(1+x))
    assert_allclose(halo.potential_gradient(R,z), [G*mass*R/r**3,G*mass*z/r**3], rtol=1e-9)


@pytest.mark.parametrize("q", [.4, .7])
def test_independent_flattened_faceon(plummer, q):
    # Swap the vertical Jeans and face-on LOS integrals analytically:
    # Sigma <vlos²> = 2 int_0^inf z nu(R,z) dPhi/dz dz.
    # Plummer potential has an exact gradient; reference uses adaptive quad.
    m = replace(plummer, tracer=PlummerTracer(300,q), inclination=0., beta_z=-.2)
    for R in [0.,100.,600.]:
        reference = 2*quad(lambda z: z*m.tracer.density(R,z)*G*1e7*z/
                           (300**2+R*R+z*z)**1.5, 0,np.inf,
                           epsabs=1e-13,epsrel=1e-9)[0]/m.tracer.surface_density(R,0,0)
        assert_allclose(m.los_second_moment(R,0),reference,rtol=2e-6)


def test_flattened_jeans_equations_and_derivative():
    m = AxisymmetricJeans(PlummerTracer(300,.65),ZhaoHalo(.1,500,Q=.7),beta_z=-.2)
    R,z,h = 220.,170.,.02
    def pressure(r,z):
        return m.tracer.density(r,z)*m.intrinsic_moments(r,z)[1]
    dR=(pressure(R+h,z)-pressure(R-h,z))/(2*h)
    dz=(pressure(R,z+h)-pressure(R,z-h))/(2*h)
    nu=m.tracer.density(R,z)
    gr,gz=m.halo.potential_gradient(R,z)
    vr,vz,vp=m.intrinsic_moments(R,z)
    assert_allclose(dz,-nu*gz,rtol=1e-7)
    assert_allclose(dR/(1-m.beta_z)+nu*(vr-vp)/R,-nu*gr,rtol=1e-7)


@pytest.mark.parametrize("Q", [.4,1.4])
def test_flattened_convergence_and_sky_symmetry(Q):
    m = AxisymmetricJeans(PlummerTracer(300,.65),ZhaoHalo(.1,500,Q=Q),
                          beta_z=-.3,inclination=.9,n_force=64,n_vertical=64,n_los=64)
    x,y = np.array([0.,30.,300.,1500.]), np.array([0.,120.,-150.,800.])
    values=m.los_second_moment(x,y)
    assert_allclose(m.los_second_moment(-x,-y),values,rtol=1e-12)
    assert_allclose(m.los_second_moment(-x,y),values,rtol=1e-12)
    fine=replace(m,n_force=128,n_vertical=128,n_los=128)
    assert_allclose(values,fine.los_second_moment(x,y),rtol=3e-4)


def test_geometry_and_domains(plummer):
    q,inc=.6,1.
    qp=PlummerTracer(300,q).projected_axis_ratio(inc)
    assert_allclose(intrinsic_axis_ratio(qp,inc),q)
    for qp,inc in [(.5,.1),(1.,0.),(1.1,1.)]:
        with pytest.raises(ValueError): intrinsic_axis_ratio(qp,inc)
    for kw in [{"beta_z":1.},{"inclination":-1.},{"n_force":12},{"n_los":32.5}]:
        with pytest.raises(ValueError): replace(plummer,**kw)
    for R,z in [(-1,0),(np.nan,0),([],[])]:
        with pytest.raises(ValueError): plummer.intrinsic_moments(R,z)
    with pytest.raises(ValueError):
        replace(plummer,beta_z=.99).intrinsic_moments(1000.,0.)
    assert np.isfinite(plummer.los_second_moment(0.,0.))
    replace(plummer, n_force=np.int64(32))
    with pytest.raises(ValueError):
        replace(plummer, n_force=np.float64(32.))


def test_existing_spherical_solver():
    from jeanspy.model import DSphModel, PlummerModel, NFWModel, ConstantAnisotropyModel
    old=DSphModel(submodels={"StellarModel":PlummerModel(),"DMModel":NFWModel(),
                            "AnisotropyModel":ConstantAnisotropyModel()})
    old.update({"re_pc":300.,"rs_pc":500.,"rhos_Msunpc3":.1,
                "r_t_pc":1e10,"beta_ani":0.,"vmem_kms":0.})
    new=AxisymmetricJeans(PlummerTracer(300),ZhaoHalo(.1,500))
    R=np.array([10.,100.,300.,1000.])
    assert_allclose(new.los_second_moment(R,0),old.sigmalos2(R),rtol=3e-4)


def test_independent_jam_equation_fixture():
    import json
    from pathlib import Path
    from jeanspy.axisymmetric import AxisymmetricDSphModel
    reference = json.loads((Path(__file__).parents[1]/"validation/axisymmetric_jam_reference.json").read_text())
    model = AxisymmetricDSphModel(96, 96, 96)
    for case in reference["cases"]:
        assert case["passed"]
        value = model.sigmalos2(reference["x_pc"], reference["y_pc"], params=case["params"])
        assert_allclose(value, case["reference"]["sigma2"],
                         rtol=reference["acceptance"]["jeanspy_vs_resolved_jam_rtol"])
