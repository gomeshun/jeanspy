"""Check that the public site contains no internal research payloads."""
from html import escape
from pathlib import Path
import argparse


RETIRED_PAGES = {
    "validation/index.html": "../theory.html",
    **{f"validation/{name}.html": "../theory.html" for name in (
        "gradients", "jam9", "los-benchmark", "performance", "release-plan", "retained-sources")},
    "comparison/index.html": "../references.html",
    "development.html": "https://github.com/gomeshun/jeanspy",
    "changelog.html": "https://github.com/gomeshun/jeanspy/releases",
    "tutorials/draco.html": "index.html",
}


def redirect_page(target):
    """Render a small noindex redirect with a visible fallback link."""
    url = escape(target, quote=True)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="robots" content="noindex">'
            '<title>JeansPy documentation</title>'
            f'<meta http-equiv="refresh" content="0; url={url}"></head>'
            '<body><p>This page has moved. '
            f'<a href="{url}">Continue</a>.</p></body></html>\n')


def check(root):
    """Fail on internal documents, assets, downloads, or search entries."""
    errors = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in RETIRED_PAGES:
            if path.read_text() != redirect_page(RETIRED_PAGES[relative]):
                errors.append(relative)
        elif any(part in path.relative_to(root).parts for part in ("validation", "comparison")):
            errors.append(relative)
        elif relative.startswith("_sources/") and any(
                name in relative for name in ("development", "changelog", "draco")):
            errors.append(relative)
        elif relative.startswith("_downloads/") and path.suffix != ".bib":
            errors.append(relative)
    search = (root / "searchindex.js").read_text()
    for name in RETIRED_PAGES:
        if '"' + name.removesuffix(".html") + '"' in search:
            errors.append("searchindex.js: " + name)
    if errors:
        raise ValueError("Internal material in public output: " + ", ".join(errors))
    return {"retired_urls": len(RETIRED_PAGES), "internal_payloads": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_dir", type=Path)
    print(check(parser.parse_args().html_dir))
