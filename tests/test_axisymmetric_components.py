"""Composed axisymmetric models preserve predictions, gradients and persistence."""
from dataclasses import FrozenInstanceError, replace
import pickle

import numpy as np
import pandas as pd
import pytest
from jeanspy.parameters import SamplingParameter

from jeanspy import axisymmetric as classical
from jeanspy._sampling_identity import fingerprint
from jeanspy.axisymmetric_inference import AxisymmetricDSphEstimationModel
from jeanspy.sampler import Sampler


PARAMS = dict(re_pc=300., q=.7, rs_pc=500., rhos_Msunpc3=.1, Q=.8,
              alpha=2., beta=3., gamma=.5, beta_z=-.3, inclination=1.1,
              r_t_pc=2000.)


def components():
    return dict(
        StellarModel=classical.AxisymmetricPlummerModel(300., .7),
        DMModel=classical.AxisymmetricZhaoModel(500., .1, .8, 2., 3., .5, 2000.),
        AnisotropyModel=classical.AxisymmetricConstantAnisotropyModel(-.3),
    )


def model():
    return classical.AxisymmetricDSphModel(16, 16, 16, submodels=components(), inclination=1.1)


def test_numpy_composition_is_immutable_and_preserves_physics():
    parts = components()
    composed = classical.AxisymmetricDSphModel(16, 16, 16, submodels=parts, inclination=1.1)
    parts["StellarModel"] = classical.AxisymmetricPlummerModel(999., .9)
    assert composed["StellarModel"].re_pc == 300.
    with pytest.raises(TypeError):
        composed.submodels["DMModel"] = parts["DMModel"]
    with pytest.raises(FrozenInstanceError):
        composed["AnisotropyModel"].beta_z = 0.
    reference = classical.AxisymmetricDSphModel(16, 16, 16)
    for method in ("sigmalos2", "density_3d", "mass_density_3d", "surface_density",
                   "intrinsic_moments", "potential_gradient"):
        np.testing.assert_allclose(getattr(composed, method)(100., 50.),
                                   getattr(reference, method)(100., 50., params=PARAMS), rtol=1e-13)
    np.testing.assert_allclose(composed.enclosed_mass([100., 3000.]),
                               reference.enclosed_mass([100., 3000.], params=PARAMS))
    for name in ("jfactor", "dfactor"):
        np.testing.assert_allclose(getattr(composed, name)(1e5, .1, n_mu=16, n_phi=16, n_radial=16),
                                   getattr(reference, name)(1e5, .1, params=PARAMS,
                                                            n_mu=16, n_phi=16, n_radial=16))
    restored = pickle.loads(pickle.dumps(composed))
    np.testing.assert_equal(restored.sigmalos2(100., 50.), composed.sigmalos2(100., 50.))


def test_component_replacement_and_call_overrides_are_effective():
    composed = model()
    original = composed.sigmalos2(100., 50.)
    changed = dict(composed.submodels)
    changed["DMModel"] = replace(changed["DMModel"], rhos_Msunpc3=.2)
    doubled = replace(composed, submodels=changed)
    np.testing.assert_allclose(doubled.sigmalos2(100., 50.), 2*original)
    np.testing.assert_allclose(composed.sigmalos2(100., 50., params={"rhos_Msunpc3": .2}),
                               2*original)
    assert composed.physical_params == PARAMS
    assert fingerprint(composed) != fingerprint(doubled)
    qp = np.sqrt(np.cos(1.1)**2 + .7**2*np.sin(1.1)**2)
    np.testing.assert_allclose(composed.sigmalos2(100., 50., params={"q_projected": qp}), original)
    with pytest.raises(ValueError, match="exactly one"):
        composed.sigmalos2(100., 50., params={"q": .7, "q_projected": qp})


@pytest.mark.parametrize("backend_name", ["classical", "jax"])
@pytest.mark.parametrize("role,base_name,methods", [
    ("StellarModel", "AxisymmetricStellarModel",
     ("density_3d", "surface_density", "radial_derivative")),
    ("DMModel", "AxisymmetricDMModel",
     ("mass_density_3d", "enclosed_mass", "potential_gradient")),
    ("AnisotropyModel", "AxisymmetricAnisotropyModel", ("beta",)),
])
def test_public_interface_alone_is_rejected_before_solver_hooks(backend_name, role, base_name, methods):
    backend = classical
    parts = components()
    if backend_name == "jax":
        pytest.importorskip("jax")
        from jeanspy import axisymmetric_jax as backend
        parts = dict(StellarModel=backend.AxisymmetricPlummerModel(),
                     DMModel=backend.AxisymmetricZhaoModel(),
                     AnisotropyModel=backend.AxisymmetricConstantAnisotropyModel())

    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError("Unsupported components must be rejected at construction")

    custom = type("CustomComponent", (getattr(backend, base_name),),
                  dict.fromkeys(methods, unexpected_evaluation))()
    assert isinstance(custom, getattr(backend, base_name))
    with pytest.raises(TypeError, match="Expected jeanspy"):
        backend.AxisymmetricDSphModel(submodels={**parts, role: custom})


def test_anisotropy_component_controls_both_meridional_and_azimuthal_moments():
    parts = components()
    old = classical._AxisymmetricJeans(parts["StellarModel"], parts["DMModel"], -.3,
                                       1.1, 16, 16, 16)
    new = classical._AxisymmetricJeans(parts["StellarModel"], parts["DMModel"],
                                       inclination=1.1, n_force=16, n_vertical=16, n_los=16,
                                       anisotropy=parts["AnisotropyModel"])
    np.testing.assert_allclose(new.intrinsic_moments(100., 50.), old.intrinsic_moments(100., 50.))


@pytest.mark.parametrize("parts,exception", [({}, ValueError),
    ({**components(), "Extra": 1}, ValueError),
    ({**components(), "AnisotropyModel": object()}, TypeError)])
def test_invalid_component_configuration_fails_early(parts, exception):
    with pytest.raises(exception):
        classical.AxisymmetricDSphModel(submodels=parts)


@pytest.mark.mcmc
def test_bound_components_supply_inference_defaults_and_restart_identity(tmp_path):
    forward = model()
    observations = dict(x_pc=[100., -100.], y_pc=[50., 80.],
                        vlos_kms=[1., -2.], e_vlos_kms=[1., 2.])
    prior = pd.DataFrame({"lower": [-1.3], "upper": [-.7]}, index=["log10_rhos_Msunpc3"])
    target = AxisymmetricDSphEstimationModel(observations, prior,
        parameter_specs=[SamplingParameter("log10_rhos_Msunpc3", "rhos_Msunpc3", "pow10")], dsph_model=forward,
                                             fixed_params={"vmem_kms": 0.})
    assert "rhos_Msunpc3" not in target.fixed_params
    assert target.fixed_params["re_pc"] == 300.
    np.testing.assert_allclose(target.convert_params([-1.])["beta_z"], -.3)
    assert np.all(np.isfinite(target.lnposterior([-1.])))
    initial = lambda n: np.array([-1.]) if n is None else np.linspace(-1.05, -.95, n)[:, None]
    first = Sampler(target, initial, nwalkers=4, prefix=str(tmp_path)+"/")
    first.run_mcmc(2, 1, enable_convergence_check=False)
    saved = first.get_chain().copy()
    same = AxisymmetricDSphEstimationModel(observations, prior,
        parameter_specs=[SamplingParameter("log10_rhos_Msunpc3", "rhos_Msunpc3", "pow10")], dsph_model=forward,
                                           fixed_params={"vmem_kms": 0.})
    resumed = Sampler(same, initial, nwalkers=4, prefix=str(tmp_path)+"/")
    resumed.run_mcmc(1, 1, enable_convergence_check=False)
    np.testing.assert_array_equal(resumed.get_chain()[:2], saved)
    parts = dict(forward.submodels)
    parts["StellarModel"] = replace(parts["StellarModel"], re_pc=350.)
    changed = AxisymmetricDSphEstimationModel(observations, prior,
        parameter_specs=[SamplingParameter("log10_rhos_Msunpc3", "rhos_Msunpc3", "pow10")],
        dsph_model=replace(forward, submodels=parts), fixed_params={"vmem_kms": 0.})
    with pytest.raises(ValueError, match="identity mismatch"):
        Sampler(changed, initial, nwalkers=4, prefix=str(tmp_path)+"/")
    assert resumed.get_chain().shape[0] == 3


def test_jax_components_and_composition_support_jit_and_gradients():
    pytest.importorskip("jax")
    from jeanspy import axisymmetric_jax as backend
    import jax
    import jax.numpy as jnp
    parts = dict(StellarModel=backend.AxisymmetricPlummerModel(),
                 DMModel=backend.AxisymmetricZhaoModel(),
                 AnisotropyModel=backend.AxisymmetricConstantAnisotropyModel())
    composed = backend.AxisymmetricDSphModel(16, 16, 16, submodels=parts)
    host = model()
    for method, role in (("density_3d", "StellarModel"), ("mass_density_3d", "DMModel"),
                         ("radial_derivative", "StellarModel"), ("beta", "AnisotropyModel")):
        value = jax.jit(lambda p: getattr(parts[role], method)(100., 50., params=p))(PARAMS)
        np.testing.assert_allclose(value, getattr(host[role], method)(100., 50.), rtol=1e-6)
    np.testing.assert_allclose(parts["StellarModel"].surface_density(100., 50., params=PARAMS),
                               host.surface_density(100., 50.), rtol=1e-6)
    result = composed.sigmalos2(100., 50., params=PARAMS)
    np.testing.assert_allclose(result, host.sigmalos2(100., 50.), rtol=1e-5)
    derivative = jax.grad(lambda rho: composed.sigmalos2(
        100., 50., params={**PARAMS, "rhos_Msunpc3": rho}))(.1)
    np.testing.assert_allclose(derivative, result/.1, rtol=1e-5)
    scale_gradient = jax.grad(lambda scale: parts["StellarModel"].density_3d(
        100., 50., params={"re_pc": scale, "q": .7}))(300.)
    finite_difference = (classical.AxisymmetricPlummerModel(300.01, .7).density_3d(100., 50.)
                         - classical.AxisymmetricPlummerModel(299.99, .7).density_3d(100., 50.))/.02
    np.testing.assert_allclose(scale_gradient, finite_difference, rtol=1e-5)
    invalid = jax.jit(lambda b: parts["AnisotropyModel"].beta(
        100., 50., params={"beta_z": b}))(jnp.array(1.))
    assert np.isnan(invalid)
    with pytest.raises(ValueError, match="Missing"):
        parts["DMModel"].enclosed_mass(100., params={"Q": .8})
    with pytest.raises(TypeError):
        backend.AxisymmetricDSphModel(submodels=components())


def test_public_axisymmetric_api_uses_canonical_components():
    assert not {"PlummerTracer", "ZhaoHalo", "AxisymmetricJeans"} & set(classical.__all__)
    for name in ("PlummerTracer", "ZhaoHalo", "AxisymmetricJeans"):
        assert not hasattr(classical, name)
    parts = components()
    assert not hasattr(parts["StellarModel"], "a_pc")
    assert not hasattr(parts["DMModel"], "rho_s")
    assert not hasattr(parts["DMModel"], "r_s")
    changed = replace(parts["DMModel"], rs_pc=600., rhos_Msunpc3=.2)
    restored = pickle.loads(pickle.dumps(changed))
    assert restored == changed
    np.testing.assert_allclose(restored.mass_density_3d(100., 50.),
                               changed.mass_density_3d(100., 50.))
