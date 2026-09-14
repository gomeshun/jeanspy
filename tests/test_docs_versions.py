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


def test_dev_replacement_removes_retired_research_payloads(tmp_path):
    build = tmp_path / "html"
    build.mkdir()
    (build / "index.html").write_text("public guide")
    site = tmp_path / "site"
    module.stage(build, site, "dev", "a" * 40)
    for name in ("validation/index.html", "_static/validation/plot.svg",
                 "_downloads/old/research.json", "_sources/comparison/index.md.txt"):
        path = site / "dev" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old internal material")
    module.stage(build, site, "dev", "b" * 40)
    assert {p.relative_to(site / "dev").as_posix() for p in (site / "dev").rglob("*")
            if p.is_file()} == {"index.html", "build-info.json"}


def test_public_boundary_allows_only_matching_quickstart_metadata(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "check_public_docs", Path(__file__).resolve().parents[1] / "scripts/check_public_docs.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    (tmp_path / "searchindex.js").write_text("Search.setIndex({})")
    original = tmp_path / "_static/quickstart/execution.json"
    original.parent.mkdir(parents=True)
    original.write_text('{"python": "3.12"}')
    download = tmp_path / "_downloads/hash/execution.json"
    download.parent.mkdir(parents=True)
    download.write_bytes(original.read_bytes())
    assert checker.check(tmp_path)["internal_payloads"] == 0
    download.write_text('{"internal": "research"}')
    with pytest.raises(ValueError, match="Internal material"):
        checker.check(tmp_path)
    download.unlink()
    notice = tmp_path / "validation/index.html"
    notice.parent.mkdir()
    notice.write_text("This page has moved")
    with pytest.raises(ValueError, match="validation/index.html"):
        checker.check(tmp_path)
