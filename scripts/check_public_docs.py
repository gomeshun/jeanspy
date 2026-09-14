"""Check that the public site contains no internal research payloads."""
from pathlib import Path
import argparse


RETIRED_PAGES = {
    "validation/index.html",
    *{f"validation/{name}.html" for name in (
        "gradients", "jam9", "los-benchmark", "performance", "release-plan", "retained-sources")},
    "comparison/index.html", "development.html", "changelog.html", "tutorials/draco.html",
}


def check(root):
    """Fail on internal documents, assets, downloads, or search entries."""
    errors = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".html":
            content = path.read_text()
            if any(marker in content for marker in (
                "/validation/", "/comparison/", "_static/validation/",
                "retained-sources", "release-plan", "CPU runtime sample",
                "JAM benchmark", "workflow validation record",
            )):
                errors.append(relative + ": internal reference")
        if relative in RETIRED_PAGES:
            errors.append(relative)
        elif any(part in path.relative_to(root).parts for part in ("validation", "comparison")):
            errors.append(relative)
        elif relative.startswith("_sources/") and any(
                name in relative for name in ("development", "changelog", "draco")):
            errors.append(relative)
        elif relative.startswith("_downloads/") and path.suffix != ".bib":
            # MyST creates a download copy for the public tutorial's execution
            # metadata. No other JSON or research payload belongs here.
            public_execution = root / "_static/quickstart/execution.json"
            if not (path.name == "execution.json" and public_execution.is_file()
                    and path.read_bytes() == public_execution.read_bytes()):
                errors.append(relative)
    search = (root / "searchindex.js").read_text()
    for name in RETIRED_PAGES:
        if '"' + name.removesuffix(".html") + '"' in search:
            errors.append("searchindex.js: " + name)
    if errors:
        raise ValueError("Internal material in public output: " + ", ".join(errors))
    return {"removed_pages": len(RETIRED_PAGES), "internal_payloads": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_dir", type=Path)
    print(check(parser.parse_args().html_dir))
