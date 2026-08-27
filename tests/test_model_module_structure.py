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
    "Exp2dModel",
    "Exp3dModel",
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
    "SimpleDSphEstimationModel",
    "StellarModel",
    "Uniform2dModel",
    "ZhaoModel",
    "get_default_estimation_model",
}


def test_public_model_api_is_explicit():
    assert set(model.__all__) == EXPECTED_PUBLIC_NAMES
    assert not hasattr(model, "np")
    assert not hasattr(model, "pd")
    assert not hasattr(model, "integrate")
    assert not hasattr(model, "SharedMemory")


def test_public_symbols_keep_historical_module_provenance():
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


def test_legacy_model_impl_is_only_a_compatibility_surface():
    compat = importlib.import_module("jeanspy._model_impl")
    assert compat.PlummerModel is model.PlummerModel
    assert compat.NFWModel is model.NFWModel
    assert compat.DSphModel is model.DSphModel
    assert compat.SersicModel is model.SersicModel


def test_ullio_private_helpers_remain_import_compatible():
    assert callable(model._ullio2016_weight)
    assert callable(model._ullio2016_inner_weight)
