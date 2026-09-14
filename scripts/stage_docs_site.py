"""Assemble versioned Sphinx output without rewriting published releases."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

from packaging.version import Version


def content_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "build-info.json":
            digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def redirect(path: Path, target: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped = html.escape(target, quote=True)
    path.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
                    '<title>JeansPy documentation</title>'
                    f'<meta http-equiv="refresh" content="0; url={escaped}">'
                    f'<a href="{escaped}">Open JeansPy documentation</a></html>\n')


def stage(html_dir: Path, site_dir: Path, version: str, commit: str,
          prerelease: bool = False, base_url: str = "https://gomeshun.github.io/jeanspy"):
    if version != "dev" and not re.fullmatch(r"v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", version):
        raise ValueError("version must be dev or an explicit vMAJOR.MINOR.PATCH release tag")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("record the full 40-character source commit")
    if not (html_dir / "index.html").is_file():
        raise ValueError("Sphinx output must contain index.html")
    prerelease = prerelease or (version != "dev" and Version(version[1:]).is_prerelease)
    info = dict(version=version, source_commit=commit, prerelease=prerelease,
                content_sha256=content_digest(html_dir))
    site_dir.mkdir(parents=True, exist_ok=True)
    target = site_dir / version
    if target.exists() and version != "dev":
        recorded = json.loads((target / "build-info.json").read_text())
        if recorded != info:
            raise ValueError(f"Published documentation {version} is immutable; refusing replacement")
    else:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(html_dir, target)
        (target / "build-info.json").write_text(json.dumps(info, indent=2) + "\n")

    releases = []
    for path in site_dir.glob("v*/build-info.json"):
        record = json.loads(path.read_text())
        releases.append(record)
    releases.sort(key=lambda row: Version(row["version"][1:]), reverse=True)
    formal = [row for row in releases if not row["prerelease"]]
    stable = formal[0]["version"] if formal else None
    entries = []
    if (site_dir / "dev").is_dir():
        entries.append(dict(name="dev (development)", version="dev", url=f"{base_url}/dev/",
                            preferred=stable is None))
    entries += [dict(name=row["version"] + (" (stable)" if row["version"] == stable else ""),
                     version=row["version"], url=f"{base_url}/{row['version']}/",
                     preferred=row["version"] == stable) for row in releases]
    (site_dir / "versions.json").write_text(json.dumps(entries, indent=2) + "\n")
    (site_dir / ".nojekyll").touch()
    if stable:
        alias = site_dir / "stable"
        if alias.exists():
            shutil.rmtree(alias)
        shutil.copytree(site_dir / stable, alias)
    redirect(site_dir / "index.html", f"{stable or 'dev'}/index.html")
    return dict(version=version, stable=stable, versions=len(entries), **{k: v for k, v in info.items() if k != "version"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-dir", required=True, type=Path)
    parser.add_argument("--site-dir", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--prerelease", action="store_true")
    parser.add_argument("--base-url", default="https://gomeshun.github.io/jeanspy")
    print(json.dumps(stage(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
