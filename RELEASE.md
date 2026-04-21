# Releasing jeanspy to PyPI with uv

This document describes the manual release flow for jeanspy. The maintainer workflow is uv-first: build with `uv build`, validate with `uvx twine`, and publish with `uv publish`. Users still install the published package with `pip install jeanspy`.

## 1. Before The First Release

- Confirm the project name is still available on PyPI: https://pypi.org/project/jeanspy/
- Create accounts on both PyPI and TestPyPI.
- Generate API tokens from the account settings pages.
- Keep the package version in `pyproject.toml` as the single source of truth.
- `jeanspy.__version__` is resolved from installed package metadata, with a local `pyproject.toml` fallback for source-tree imports.

## 2. Prepare The Release

1. Update the version in `pyproject.toml`.

For an explicit version:

```bash
uv version 0.1.1
```

For a semantic bump:

```bash
uv version --bump patch
```

2. Review `README.md` and `pyproject.toml` for any release-specific changes.
3. Remove old build artifacts so only the current release files remain in `dist/`.

```bash
rm -rf dist/
```

4. Optionally create a Git tag or release branch before uploading.

## 3. Build And Validate

From the repository root:

```bash
uv build --no-sources
uvx twine check dist/*
uv run pytest tests/test_import.py -q
```

- `uv build --no-sources` is recommended for publishing so the build is closer to what downstream tools see outside your local uv project.
- If you changed core behavior, run additional targeted tests before uploading.

Optional smoke test of the built wheel in an isolated environment:

```bash
uv venv .venv-release-test --python 3.12
. .venv-release-test/bin/activate
uv pip install dist/jeanspy-<version>-py3-none-any.whl
python -c "import jeanspy; print(jeanspy.__version__)"
deactivate
rm -rf .venv-release-test
```

## 4. Upload To TestPyPI

Use a TestPyPI token first so you can verify the package metadata and installation flow without consuming the real release version on PyPI.

```bash
export UV_PUBLISH_TOKEN=<testpypi-token>
uv publish \
  --publish-url https://test.pypi.org/legacy/ \
  --check-url https://test.pypi.org/simple/
```

If the upload fails partway through, rerun the same command. uv skips files that already exist and match exactly.

Then verify installation from TestPyPI in a clean environment:

```bash
uv venv .venv-testpypi --python 3.12
. .venv-testpypi/bin/activate
uv pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  jeanspy
python -c "import jeanspy; print(jeanspy.__version__)"
deactivate
rm -rf .venv-testpypi
```

## 5. Upload To PyPI

When the TestPyPI upload looks correct, publish the same artifacts to PyPI:

```bash
export UV_PUBLISH_TOKEN=<pypi-token>
uv publish
```

If you switch to Trusted Publishing later, `UV_PUBLISH_TOKEN` is no longer needed.

## 6. Post-Release Checks

- Install from PyPI in a clean environment: `uv run --with jeanspy --no-project -- python -c "import jeanspy; print(jeanspy.__version__)"`
- Open the PyPI project page and confirm the README, classifiers, and links render correctly.
- Create a GitHub release if you want tags and release notes to match the published version.

## 7. Recommended Follow-Up

For repeatable releases, consider configuring PyPI Trusted Publishing from GitHub Actions so that you can publish without storing long-lived API tokens.