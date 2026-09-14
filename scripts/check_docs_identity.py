"""Check the version and copyable install commands in rendered documentation."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.links, self.text, self.commands = [], [], []
        self.hidden = 0
        self.in_pre = False
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag == "pre":
            self.in_pre = True
        if tag == "a":
            self.links.append(dict(attrs).get("href"))

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1
        if tag == "pre":
            self.in_pre = False

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)
            if self.in_pre:
                self.commands.append(data)


def check(directory, version, source_ref):
    home = Page(directory / "index.html")
    install = Page(directory / "installation.html")
    prose = re.sub(r"\s+", " ", "".join(home.text))
    commands = "".join(install.commands)
    if f"https://github.com/gomeshun/jeanspy/tree/{source_ref}" not in home.links:
        raise ValueError("Home page does not link to the requested source reference")
    if f"Documentation version: {version}." not in prose:
        raise ValueError("Home page does not identify the requested documentation version")
    if f"git checkout {source_ref}\n" not in commands:
        raise ValueError("Source installation does not check out the requested reference")
    if re.search(r"@(DOCS_VERSION|PACKAGE_VERSION|SOURCE_REF|SOURCE_LABEL)@",
                 "".join(home.text + install.text)):
        raise ValueError("An unresolved build-identity token is visible")
    if version == "dev":
        if "development documentation" not in prose or "jeanspy==" in commands:
            raise ValueError("Development documentation has release installation instructions")
    else:
        package_version = version.removeprefix("v")
        if "development documentation" in prose:
            raise ValueError("Release home page still identifies itself as development")
        for extra in ("", "[numpyro_cpu,plotting]", "[numpyro_cuda12,plotting]"):
            if f"python -m pip install 'jeanspy{extra}=={package_version}'" not in commands:
                raise ValueError(f"Missing version-pinned installation command for {extra or 'base'}")
    return dict(version=version, source_ref=source_ref, rendered_identity="passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.directory, args.version, args.source_ref), indent=2))
