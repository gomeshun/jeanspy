import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import jeanspy

def test_version_available():
    assert hasattr(jeanspy, "__version__")


def test_sampler_all_contains_only_sampler():
    from jeanspy import sampler
    assert sampler.__all__ == ["Sampler"]


def test_sampler_wildcard_import_no_legacy_names():
    from jeanspy import sampler
    for name in ("DSphSimulator", "Network", "Worker", "_LEGACY_SWYFT_EXPORTS"):
        assert name not in sampler.__all__, f"{name} should not be in sampler.__all__"
        assert not hasattr(sampler, name), f"{name} should not be accessible on sampler module"


def test_sampler_class_importable():
    from jeanspy.sampler import Sampler
    assert Sampler is not None
