import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import jeanspy


def test_version_available():
    assert hasattr(jeanspy, "__version__")


def test_public_modules_importable():
    from jeanspy import dequad, jfactor, model, sampler


def test_package_public_api_boundary():
    assert jeanspy.__all__ == ["dequad", "model", "jfactor", "sampler"]

    historical_modules = {"coord", "polygon", "cmd_utilities"}
    assert historical_modules.isdisjoint(jeanspy.__all__)
    assert historical_modules.isdisjoint(jeanspy._LAZY_MODULES)


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
