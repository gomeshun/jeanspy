"""Regression tests for the classical model module split (issue #16)."""

import importlib
import pickle

import pytest

import jeanspy.model as model


EXPECTED_PUBLIC_NAMES = {
    "AnisotropyModel",
    "BaesAnisotropyModel",
    "C_J",
    "ConstantAnisotropyModel",
    "DMModel",
    "DSphModel",
    "DotDict",
    "ProjectedExponentialModel",
    "FittableModel",
    "FlatPriorModel",
    "GMsun_m3s2",
    "Model",
    "NFWModel",
    "OsipkovMerrittModel",
    "Parameters",
    "PhotometryPriorModel",
    "PlummerModel",
    "SersicModel",
    "SphericalDSphEstimationModel",
    "StellarModel",
    "Uniform2dModel",
    "ZhaoModel",
    "plummer_nfw_constant_anisotropy_model",
}


def test_public_model_api_is_explicit():
    assert set(model.__all__) == EXPECTED_PUBLIC_NAMES | {
        "AxisymmetricDSphModel", "AxisymmetricDSphEstimationModel", "AxisymmetricKinematicData",
        "AxisymmetricStellarModel", "AxisymmetricPlummerModel", "AxisymmetricDMModel",
        "AxisymmetricZhaoModel", "AxisymmetricAnisotropyModel",
        "AxisymmetricConstantAnisotropyModel"}
    assert not hasattr(model, "np")
    assert not hasattr(model, "pd")
    assert not hasattr(model, "integrate")
    assert not hasattr(model, "SharedMemory")


def test_public_symbols_use_the_supported_module_path():
    for name in EXPECTED_PUBLIC_NAMES:
        value = getattr(model, name)
        if hasattr(value, "__module__"):
            assert value.__module__ == "jeanspy.model", name


def test_parameters_remain_pickleable_through_public_module():
    parameters = model.Parameters({"x": 1.5})
    restored = pickle.loads(pickle.dumps(parameters))
    assert isinstance(restored, model.Parameters)
    assert restored["x"] == 1.5


def test_sersic_uses_split_stellar_base():
    assert issubclass(model.SersicModel, model.StellarModel)
    assert model.SersicModel.__module__ == "jeanspy.model"


def test_dotdict_missing_attribute_uses_requested_name():
    values = model.DotDict({"present": 1})
    with pytest.raises(AttributeError, match="^missing$"):
        _ = values.missing


@pytest.mark.parametrize("name", ["jeanspy._model_impl", "jeanspy._classical"])
def test_retired_internal_modules_are_removed(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_facade_does_not_reexport_private_factor_helpers():
    assert not hasattr(model, "_ullio2016_weight")
    assert not hasattr(model, "_ullio2016_inner_weight")
