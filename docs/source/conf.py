"""English documentation for the checked-out JeansPy source."""
from pathlib import Path
import os
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JEANSPY_JAX_PLATFORM", "cpu")
os.environ.setdefault("JEANSPY_JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jeanspy-docs-mpl")

project = "JeansPy"
author = "Shunichi Horigome"
copyright = "2026, Shunichi Horigome"
release = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
version = os.environ.get("JEANSPY_DOCS_VERSION", "dev")
if version != "dev" and (
    not re.fullmatch(r"v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", version)
    or version != f"v{release}"
):
    raise ValueError("Release documentation tag must match the package version")
source_ref = os.environ.get("JEANSPY_DOCS_REF", "main" if version == "dev" else version)
jeanspy_docs_source_ref = source_ref
if not re.fullmatch(r"[A-Za-z0-9_./-]+", source_ref):
    raise ValueError("Documentation source reference contains unsupported characters")
tags.add("development" if version == "dev" else "release")
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
# Research records remain in the repository, outside the published site.
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store", "validation/**", "comparison/**",
    "_static/validation/**", "changelog.md", "development.md", "tutorials/draco.md",
]
templates_path = ["_templates"]
autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "fully-qualified"
autodoc_type_aliases = {"DSphModel": "jeanspy.model_numpyro.DSphModel"}
autodoc_class_signature = "mixed"
napoleon_use_rtype = False
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 4
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
intersphinx_cache_limit = 7
html_theme = "pydata_sphinx_theme"
html_title = "JeansPy documentation"
html_static_path = ["_static"]
html_css_files = ["jeanspy.css"]
html_theme_options = {
    "github_url": "https://github.com/gomeshun/jeanspy",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": os.environ.get("JEANSPY_DOCS_SWITCHER", "../versions.json"),
        "version_match": version,
    },
    "check_switcher": False,
    # A stable alias does not exist before the first formal release. The
    # default banner incorrectly labels the only preferred entry as stable.
    "show_version_warning_banner": False,
    "navigation_with_keys": False,
    "header_links_before_dropdown": 6,
    "footer_start": ["copyright"],
}
html_context = {
    "default_mode": "auto",
    "github_user": "gomeshun", "github_repo": "jeanspy",
    "github_version": source_ref,
    "doc_path": "docs/source",
}
html_baseurl = f"https://gomeshun.github.io/jeanspy/{version}/"
html_last_updated_fmt = "%Y-%m-%d"
doctest_global_setup = "import numpy as np"


def substitute_build_identity(app, docname, source):
    """Insert build identity in prose and copyable commands before parsing.

    The small explicit token set also works inside Markdown code fences, where
    MyST's ordinary substitutions are deliberately not evaluated. Keep the
    surrounding instructions in the Markdown pages rather than in this hook.
    """
    if docname not in {"index", "installation"}:
        return
    for token, value in {
        "@DOCS_VERSION@": version,
        "@PACKAGE_VERSION@": release,
        "@SOURCE_REF@": source_ref,
        "@SOURCE_LABEL@": source_ref[:12] if re.fullmatch(r"[0-9a-f]{40}", source_ref) else source_ref,
    }.items():
        source[0] = source[0].replace(token, value)


def qualify_imported_types(app, doctree):
    """Resolve the sampler's imported DSphModel to its actual JAX definition.

    Sphinx 9's constructor type renderer retains the unqualified annotation
    even with fully-qualified output enabled. There are two public DSphModel
    classes, so preserve the import's meaning explicitly instead of accepting
    Sphinx's ambiguous fallback to the classical class.
    """
    from sphinx import addnodes
    for node in doctree.findall(addnodes.pending_xref):
        if (node.get("py:module") == "jeanspy.sampler_numpyro"
                and node.get("reftarget") == "DSphModel"):
            node["reftarget"] = "jeanspy.model_numpyro.DSphModel"


def preserve_source_alias_anchors(app, doctree):
    """Keep viewcode's return links valid for the public convenience alias.

    Sphinx viewcode keeps one reference-module name per source module. This
    module also documents native objects, so its return link can use the
    implementation name although the class is documented through model.
    Both names point to the same runtime class; add the corresponding anchors.
    """
    from sphinx import addnodes
    canonical = "jeanspy.model.AxisymmetricDSphModel"
    implementation = "jeanspy.axisymmetric.AxisymmetricDSphModel"
    for node in doctree.findall(addnodes.desc_signature):
        for target in list(node.get("ids", [])):
            if target == canonical or target.startswith(canonical + "."):
                alias = implementation + target[len(canonical):]
                if alias not in node["ids"]:
                    node["ids"].append(alias)


def setup(app):
    # The source reference appears inside parsed prose/code blocks, so changing
    # it must invalidate cached doctrees, not only the HTML template context.
    app.add_config_value("jeanspy_docs_source_ref", "main", "env")
    app.connect("source-read", substitute_build_identity)
    app.connect("doctree-read", qualify_imported_types)
    app.connect("doctree-read", preserve_source_alias_anchors)
    app.connect("build-finished", retire_research_pages)


def retire_research_pages(app, exception):
    """Keep old bookmarks usable without publishing research pages or data."""
    if exception is not None or app.builder.name != "html":
        return
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_public_docs import RETIRED_PAGES, redirect_page
    for old, target in RETIRED_PAGES.items():
        path = Path(app.outdir) / old
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(redirect_page(target))
