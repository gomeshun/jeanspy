"""An otherwise usable sdist must not silently omit documentation sources."""
import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "check_distribution_contents", Path(__file__).resolve().parents[1] / "scripts/check_distribution_contents.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("missing", ["docs/source/conf.py", "docs/source/references.bib",
                                     "docs/source/_templates/autosummary/class.rst",
                                     "docs/source/_static/validation/figure.svg"])
def test_missing_documentation_rejected(tmp_path, monkeypatch, missing):
    source = tmp_path / "source"
    names = ["src/jeanspy/__init__.py", "README.md", "RELEASE.md", "LICENSE",
             "pyproject.toml", "MANIFEST.in", "validation/axisymmetric_jam_reference.json",
             "validation/release/jam9_protocol.json", "validation/release/jam9_plummer_v1.json", missing]
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test source\n")
    # Sphinx output and the generated API are intentionally absent from sdists.
    for name in ("docs/_build/html/conf.py", "docs/source/api/generated/class.rst"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded output")
    distribution = tmp_path / "dist"
    distribution.mkdir()
    with zipfile.ZipFile(distribution / "jeanspy.whl", "w") as archive:
        archive.write(source / "src/jeanspy/__init__.py", "jeanspy/__init__.py")
    def write_source(omit):
        with tarfile.open(distribution / "jeanspy.tar.gz", "w:gz") as archive:
            for name in names:
                if name == omit:
                    continue
                content = (source / name).read_bytes()
                member = tarfile.TarInfo("jeanspy/" + name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    monkeypatch.setattr(module, "ROOT", source)
    write_source(None)
    assert len(module.check(distribution)) == 2
    write_source(missing)
    with pytest.raises(ValueError, match="missing " + missing):
        module.check(distribution)
