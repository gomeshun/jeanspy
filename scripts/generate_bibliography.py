"""Render BibTeX and a readable bibliography from reviewed citation metadata."""
from pathlib import Path
from html import unescape
import json

ROOT = Path(__file__).resolve().parents[1]


def tex(text):
    replacements = {
        "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
        "γ": r"$\gamma$", "ν": r"$\nu$", "β": r"$\beta$",
        "ω": r"$\omega$", "–": "--", "—": "---", " ": " ",
        "ö": r'{\"o}', "ü": r'{\"u}', "ä": r'{\"a}',
        "é": r"{\'e}", "è": r"{\`e}", "á": r"{\'a}",
        "í": r"{\'i}", "ó": r"{\'o}", "ñ": r"{\~n}",
    }
    return "".join(replacements.get(char, char) for char in unescape(str(text)))


def main():
    data = json.loads((ROOT / "docs/references.json").read_text())
    entries = data["references"]
    bib = ["% Generated from docs/references.json. Do not edit by hand.", ""]
    page = ["# References", "", "References are grouped by their role in an analysis. Cite the methods you use",
            "together with the version of JeansPy and its dependencies.", "",
            "Download the {download}`BibTeX file <references.bib>`.", ""]
    category = None
    for entry in entries:
        authors = entry["authors"]
        fields = {"author": " and ".join(tex(a["family"] + ", " + a.get("given", "")) for a in authors),
                  "title": "{" + tex(entry["title"]) + "}", "year": str(entry["year"])}
        for key in ("journal", "volume", "pages", "doi", "url", "eprint", "archivePrefix", "note", "publisher", "edition"):
            if entry.get(key):
                fields[key] = str(entry[key]) if key in ("url", "doi") else tex(entry[key])
        if "pages" in fields:
            fields["pages"] = fields["pages"].replace("-", "--")
        bib += ["@" + entry["entry_type"] + "{" + entry["key"] + ","]
        bib += [f"  {key} = {{{value}}}," for key, value in fields.items()]
        bib += ["}", ""]
        if entry["category"] != category:
            category = entry["category"]
            anchor = category.lower().replace(" ", "-")
            page += [f"(references-{anchor})=", f"## {category}", "",
                     "| Reference | Work | Relevance |", "| --- | --- | --- |"]
        names = " & ".join(a["family"] for a in authors)
        if len(authors) > 2:
            names = authors[0]["family"] + " et al."
        title = unescape(entry["title"]).replace("|", r"\|")
        page += [f'| <span id="reference-{entry["key"]}"></span>{names} ({entry["year"]}) '
                 f'| [{title}]({entry["url"]}) | {entry["role"]} |']
        if entry is entries[-1] or entries[entries.index(entry)+1]["category"] != category:
            page.append("")
    (ROOT / "docs/source/references.bib").write_text("\n".join(bib))
    (ROOT / "docs/source/references.md").write_text("\n".join(page))
    print(f"Rendered {len(entries)} references")


if __name__ == "__main__":
    main()
