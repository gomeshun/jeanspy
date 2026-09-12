"""JAX parity and automatic differentiation of the actual nested solver."""
from dataclasses import replace
from contextlib import contextmanager
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jeanspy.axisymmetric import AxisymmetricDSphModel as NumpyModel
from jeanspy.axisymmetric_numpyro import AxisymmetricDSphModel as JaxModel

P = dict(re_pc=300., rs_pc=500., rhos_Msunpc3=.1, q=.65, Q=.7,
         alpha=2., beta=3., gamma=.8, beta_z=-.3, inclination=1.1)


@contextmanager
def x64(enabled):
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


@pytest.fixture(autouse=True)
def precision():
    with x64(True):
        yield


@pytest.mark.parametrize("Q,inc", [(.4,0.),(.7,1.1),(1.4,np.pi/2)])
def test_backend_parity(Q,inc):
    p={**P,"Q":Q,"inclination":inc}
    a,b=NumpyModel(48,48,48),JaxModel(48,48,48)
    x,y=np.array([0.,50.,300.]),np.array([0.,-100.,200.])
    for name,args in [("sigmalos2",(x,y)),("intrinsic_moments",(x,y)),
                      ("potential_gradient",(x[1:],y[1:])),("surface_density",(x,y))]:
        np.testing.assert_allclose(getattr(a,name)(*args,params=p),getattr(b,name)(*args,params=p),rtol=2e-10,atol=1e-12)


def test_all_parameter_gradients_and_jit():
    model=JaxModel(32,32,32)
    def prediction(p):
        return jnp.sum(model.sigmalos2(jnp.array([50.,200.]),jnp.array([80.,-100.]),params=p))
    gradients=jax.jit(jax.grad(prediction))(P)
    for name,value in P.items():
        h=1e-4*max(abs(value),.1)
        fd=(prediction({**P,name:value+h})-prediction({**P,name:value-h}))/(2*h)
        assert np.isfinite(gradients[name]),name
        np.testing.assert_allclose(gradients[name],fd,rtol=3e-5,atol=1e-6,err_msg=name)
    # Exact normalization scaling, including the derivative through all integrals.
    np.testing.assert_allclose(gradients['rhos_Msunpc3'],prediction(P)/P['rhos_Msunpc3'],rtol=1e-11)


def test_projected_flattening_gradient():
    p={k:v for k,v in P.items() if k!='q'}
    p['q_projected']=.8
    m=JaxModel(32,32,32)
    f=lambda inc:m.sigmalos2(50.,100.,params={**p,'inclination':inc})
    h=1e-4
    np.testing.assert_allclose(jax.grad(f)(1.1),(f(1.1+h)-f(1.1-h))/(2*h),rtol=1e-5)
    np.testing.assert_allclose(f(1.1),NumpyModel(32,32,32).sigmalos2(50.,100.,params=p),rtol=1e-10)


@pytest.mark.parametrize('changes',[{'q':-1.},{'beta_z':1.},{'rhos_Msunpc3':np.nan},
                                    {'gamma':2.},{'inclination':-1.},{'beta_z':.99}])
def test_invalid_proposals(changes):
    assert np.isnan(JaxModel(24,24,24).sigmalos2(500.,0.,params={**P,**changes}))


def test_shapes_vmap_and_axis_gradients():
    m=JaxModel(24,24,24)
    p={**P,'inclination':0.}
    gradients=jax.grad(lambda q:m.sigmalos2(0.,0.,params={**p,'q':q}))(.65)
    assert np.isfinite(gradients)
    value=jax.vmap(lambda rho:m.sigmalos2(100.,50.,params={**P,'rhos_Msunpc3':rho}))(
        jnp.array([.1,.2]))
    np.testing.assert_allclose(value[1],2*value[0],rtol=1e-12)
    assert m.sigmalos2(jnp.ones((2,1)),jnp.ones(3),params=P).shape==(2,3)
    assert np.isnan(m.sigmalos2(jnp.nan,0.,params=P))


def test_float32_parity():
    with x64(False):
        m=JaxModel(48,48,48)
        value=m.sigmalos2(jnp.array([0.,100.,1000.]),0.,params=P)
        assert value.dtype==jnp.float32
        np.testing.assert_allclose(value,NumpyModel(48,48,48).sigmalos2([0.,100.,1000.],0.,params=P),rtol=3e-5)


@pytest.mark.parametrize("Q", [.45, 1., 1.7])
def test_finite_halo_backend_parity_and_gradients(Q):
    p = {**P, "Q": Q, "r_t_pc": 600.}
    a, b = NumpyModel(48, 64, 64), JaxModel(48, 64, 64)
    x, y = np.array([0., 100., 800.]), np.array([0., 50., 900.])
    for name, args in [("potential_gradient", (x[1:], y[1:])),
                       ("intrinsic_moments", (x, y)), ("sigmalos2", (x, y)),
                       ("density_3d", (x, y)), ("mass_density_3d", (x, y)),
                       ("enclosed_mass", (np.array([0., 500., 900., np.inf]),))]:
        np.testing.assert_allclose(getattr(a, name)(*args, params=p),
                                   getattr(b, name)(*args, params=p), rtol=3e-10, atol=1e-12)
    f = lambda rt: b.sigmalos2(700., 400., params={**p, "r_t_pc": rt})
    h = .01
    np.testing.assert_allclose(jax.grad(f)(600.), (f(600.+h)-f(600.-h))/(2*h), rtol=3e-4, atol=1e-7)


def test_core_origin_force_coordinate_derivative():
    from jeanspy.axisymmetric import G
    p = {**P, "Q": 1., "alpha": 2., "beta": 5., "gamma": 0.}
    m = JaxModel(32, 32, 32)
    gradient = jax.jacfwd(lambda x: jnp.array(m.potential_gradient(x[0], x[1], params=p)))(jnp.zeros(2))
    np.testing.assert_allclose(gradient, np.eye(2)*4*np.pi*G*p["rhos_Msunpc3"]/3, rtol=1e-12)


def test_nearly_faceon_round_photometry_deprojection():
    from jeanspy.axisymmetric import intrinsic_axis_ratio
    p = {k: v for k, v in P.items() if k != "q"}
    p.update(q_projected=1., inclination=1e-9)
    assert intrinsic_axis_ratio(1., 1e-9) == 1.
    np.testing.assert_allclose(JaxModel(24, 24, 24).sigmalos2(30., 10., params=p),
                               NumpyModel(24, 24, 24).sigmalos2(30., 10., params=p), rtol=1e-11)
