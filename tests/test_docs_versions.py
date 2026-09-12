"""Published documentation must survive dev and prerelease deployments."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "stage_docs_site", Path(__file__).resolve().parents[1] / "scripts/stage_docs_site.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_release_preservation_and_stable_selection(tmp_path):
    build = tmp_path / "html"
    build.mkdir()
    (build / "index.html").write_text("development")
    site = tmp_path / "site"
    module.stage(build, site, "dev", "a" * 40)
    assert not (site / "stable").exists()
    module.stage(build, site, "v0.1.0rc1", "b" * 40, prerelease=True)
    assert not (site / "stable").exists()
    (build / "index.html").write_text("release")
    module.stage(build, site, "v0.1.0", "c" * 40)
    assert (site / "stable/index.html").read_text() == "release"
    (build / "index.html").write_text("next development")
    module.stage(build, site, "dev", "d" * 40)
    assert (site / "v0.1.0/index.html").read_text() == "release"
    assert (site / "stable/index.html").read_text() == "release"
    with pytest.raises(ValueError, match="immutable"):
        module.stage(build, site, "v0.1.0", "d" * 40)
    module.stage(build, site, "v0.0.9", "e" * 40)
    assert (site / "stable/index.html").read_text() == "release"
