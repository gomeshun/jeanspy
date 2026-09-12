"""Render BibTeX and a readable bibliography from reviewed citation metadata."""
from pathlib import Path
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
    return "".join(replacements.get(char, char) for char in str(text))


def main():
    data = json.loads((ROOT / "validation/release/references.json").read_text())
    entries = data["references"]
    bib = ["% Generated from validation/release/references.json. Do not edit by hand.", ""]
    page = ["# References", "", "This list covers scientific methods, comparison papers, data and inference",
            "software used in these documents. Cite only the methods and resources",
            "used in a particular analysis, together with their actual code versions.", "",
            "Download the {download}`BibTeX file <references.bib>` or the",
            "{download}`reviewed metadata <../../validation/release/references.json>`.", ""]
    for entry in entries:
        authors = entry["authors"]
        fields = {"author": " and ".join(tex(a["family"] + ", " + a.get("given", "")) for a in authors),
                  "title": "{" + tex(entry["title"]) + "}", "year": str(entry["year"])}
        for key in ("journal", "volume", "pages", "doi", "url", "eprint", "archivePrefix", "note"):
            if entry.get(key):
                fields[key] = str(entry[key]) if key in ("url", "doi") else tex(entry[key])
        if "pages" in fields:
            fields["pages"] = fields["pages"].replace("-", "--")
        bib += ["@" + entry["entry_type"] + "{" + entry["key"] + ","]
        bib += [f"  {key} = {{{value}}}," for key, value in fields.items()]
        bib += ["}", ""]
        names = ", ".join(a["family"] for a in authors[:4])
        if len(authors) > 4:
            names = authors[0]["family"] + " et al."
        location = ", ".join(str(entry[k]) for k in ("journal", "volume", "pages") if entry.get(k))
        page += [f"(reference-{entry['key']})=", f"## {names} ({entry['year']})", "",
                 f"[{entry['title']}]({entry['url']})." + (f" {location}." if location else ""), ""]
    (ROOT / "docs/source/references.bib").write_text("\n".join(bib))
    (ROOT / "docs/source/references.md").write_text("\n".join(page))
    print(f"Rendered {len(entries)} references")


if __name__ == "__main__":
    main()
