"""Check that PyPI archives contain the runtime and a usable source test suite.

Run after ``uv build --no-sources``. This opens archives without extracting or
executing their contents. The wheel is deliberately limited to the package;
the sdist also carries fixtures, examples and notebook sources used by tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def check(directory: Path) -> list[dict]:
    wheels = sorted(directory.glob("*.whl"))
    sources = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Use a clean build directory with exactly one wheel and one sdist")
    runtime = {p.relative_to(ROOT / "src").as_posix(): p
               for p in (ROOT / "src/jeanspy").rglob("*")
               if p.is_file() and p.suffix in {".py", ".csv"}}
    support = {p.relative_to(ROOT).as_posix(): p
               for base, suffix in [("tests", ".py"), ("examples", ".py"),
                                    ("notebooks", ".ipynb"), ("scripts", ".py")]
               for p in (ROOT / base).rglob("*" + suffix)}
    # Mirror the intended documentation source payload in MANIFEST.in. Build
    # output and the regenerated API tree must not affect an archive check.
    docs_suffixes = {".md", ".rst", ".py", ".css", ".json", ".bib", ".svg", ".png", ".pdf"}
    support.update({p.relative_to(ROOT).as_posix(): p
                    for p in (ROOT / "docs").rglob("*")
                    if p.is_file() and p.suffix in docs_suffixes
                    and not p.is_relative_to(ROOT / "docs/_build")
                    and not p.is_relative_to(ROOT / "docs/source/api")
                    and "__pycache__" not in p.parts})
    support.update({name: ROOT / name for name in
                    ["README.md", "RELEASE.md", "LICENSE", "pyproject.toml", "MANIFEST.in",
                     "validation/axisymmetric_jam_reference.json",
                     "validation/release/jam9_protocol.json",
                     "validation/release/jam9_plummer_v1.json"]})
    reports = []
    for artifact in wheels + sources:
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                contents = {name: archive.read(name) for name in archive.namelist()
                            if not name.endswith("/")}
            expected = runtime
            if any(not (name.startswith("jeanspy/") or ".dist-info/" in name)
                   for name in contents):
                raise ValueError("Wheel contains files outside jeanspy and its metadata")
        else:
            with tarfile.open(artifact) as archive:
                members = [m for m in archive.getmembers() if not m.isdir()]
                if any(not m.isfile() for m in members):
                    raise ValueError("Source distribution contains a non-regular file")
                contents = {m.name.partition("/")[2]: archive.extractfile(m).read()
                            for m in members}
            expected = {"src/" + name: path for name, path in runtime.items()} | support
        for name in contents:
            parts = PurePosixPath(name).parts
            if (".." in parts or PurePosixPath(name).is_absolute()
                    or any(v in parts for v in ["external", "jeanspy_paper", "__pycache__", "_demo_outputs"])
                    or name.endswith((".pyc", ".pkl", ".h5", ".nc"))):
                raise ValueError(f"Unexpected distribution member: {name}")
        for name, source in expected.items():
            if name not in contents:
                raise ValueError(f"{artifact.name} is missing {name}")
            if contents[name] != source.read_bytes():
                raise ValueError(f"{artifact.name} has stale or changed content in {name}")
        reports.append({"artifact": artifact.name, "bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "files": len(contents), "verified_source_files": len(expected)})
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = check(args.directory)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    print(rendered, end="")
