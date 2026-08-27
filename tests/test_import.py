import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import jeanspy


def test_version_available():
    assert hasattr(jeanspy, "__version__")


def test_core_modules_importable():
    from jeanspy import cmd_utilities, coord, dequad, jfactor, model, polygon


def test_sampler_all_contains_only_sampler():
    from jeanspy import sampler
    assert sampler.__all__ == ["Sampler"]


def test_sampler_wildcard_import_no_legacy_names():
    namespace = {}
    exec("from jeanspy.sampler import *", namespace)

    assert "Sampler" in namespace
    for name in ("DSphSimulator", "Network", "Worker", "_LEGACY_SWYFT_EXPORTS"):
        assert name not in namespace, f"{name} should not be imported from jeanspy.sampler"


def test_sampler_legacy_names_not_accessible():
    from jeanspy import sampler

    for name in ("DSphSimulator", "Network", "Worker", "_LEGACY_SWYFT_EXPORTS"):
        assert not hasattr(sampler, name), f"{name} should not be accessible on sampler module"


def test_sampler_class_importable():
    from jeanspy.sampler import Sampler
    assert Sampler is not None


def test_model_public_exports_are_explicit():
    from jeanspy import model

    expected = {
        "Parameters",
        "Model",
        "StellarModel",
        "PlummerModel",
        "SersicModel",
        "Exp2dModel",
        "Exp3dModel",
        "Uniform2dModel",
        "DMModel",
        "ZhaoModel",
        "NFWModel",
        "AnisotropyModel",
        "ConstantAnisotropyModel",
        "OsipkovMerrittModel",
        "BaesAnisotropyModel",
        "DSphModel",
        "FittableModel",
        "FlatPriorModel",
        "PhotometryPriorModel",
        "SimpleDSphEstimationModel",
        "DotDict",
        "KI17_Model",
        "get_default_estimation_model",
        "C_J",
    }
    assert set(model.__all__) == expected

    namespace = {}
    exec("from jeanspy.model import *", namespace)
    assert {name for name in namespace if not name.startswith("_")} == expected


def test_model_facade_does_not_leak_implementation_names():
    from jeanspy import model

    for name in ("_impl", "np", "pd", "multi", "SharedMemory", "DATA_DIR"):
        assert not hasattr(model, name), f"{name} should remain private"
