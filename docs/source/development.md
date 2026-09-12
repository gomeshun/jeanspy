# Building and maintaining the documentation

Source pages live in `docs/source`; earlier audit reports remain in `docs`.
`scripts/generate_api_docs.py` inventories supported modules and creates
autosummary pages from the checked-out source. Sphinx imports the real optional
backend, rather than mocking JAX or NumPyro. Short executable examples are
included directly from Python files and exercised by CI.

Build with the commands in [installation](installation.md). Warnings fail the
build. Internal links, search, equations, figures and the version selector
must also be checked in a browser before publication. PR builds retain HTML
as Actions artifacts. Main publishes `dev`; published GitHub releases add
immutable version directories, and the first formal release establishes
`stable`. No package release is created by building the documentation.

The site uses [Sphinx autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html),
[MyST math syntax](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html),
the [PyData version switcher](https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html)
and [GitHub Pages deployment workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
