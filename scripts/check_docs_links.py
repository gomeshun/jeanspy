"""Check generated HTML files and anchors, including viewcode return links."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if tag == "a":
            if "name" in attributes:
                self.ids.add(attributes["name"])
            if "href" in attributes:
                self.links.append(attributes["href"])


def check(root: Path) -> dict:
    root = root.resolve()
    pages = {}
    for path in root.rglob("*.html"):
        page = Page()
        page.feed(path.read_text())
        pages[path] = page
    if not pages:
        raise ValueError("No built HTML pages found")
    errors, count = [], 0
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not (url.path or url.fragment):
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            count += 1
            reason = None
            if not target.exists():
                reason = "missing file"
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                reason = "missing anchor"
            if reason:
                errors.append(dict(page=str(path.relative_to(root)), link=link, reason=reason))
    return dict(html_pages=len(pages), local_links_checked=count, broken_links=errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = check(args.html_dir)
    content = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(content)
    print(content, end="")
    raise SystemExit(bool(result["broken_links"]))
