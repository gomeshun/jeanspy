# Releasing jeanspy to PyPI

JeansPy is published from GitHub Actions with `uv` and PyPI Trusted Publishing. The release workflow is `.github/workflows/release.yml`.

No long-lived PyPI API token is stored in GitHub. A version tag triggers a build and validation job; only the resulting artifacts are passed to a separate publishing job with GitHub OIDC permission.

Users install the published package with:

```bash
pip install jeanspy
```

## 1. One-Time Setup Before The First Release

### Create the GitHub environment

In the `gomeshun/jeanspy` repository, open:

`Settings` -> `Environments` -> `New environment`

Create an environment named:

```text
pypi
```

No PyPI secret is required. Optionally configure required reviewers on this environment if you want every release to require manual approval before the publish job runs.

### Register the PyPI Trusted Publisher

If `jeanspy` does not yet exist on PyPI, add a **pending publisher** from the PyPI account's `Publishing` page. Use:

| Field | Value |
| --- | --- |
| PyPI project name | `jeanspy` |
| GitHub owner | `gomeshun` |
| Repository | `jeanspy` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

A pending publisher creates the PyPI project on the first successful upload. It does not reserve the project name before that upload.

If the project already exists on PyPI, configure the same Trusted Publisher from that project's `Publishing` settings instead.

## 2. Prepare A Release

Keep the package version in `pyproject.toml` as the single source of truth. The Git tag must match it exactly with a leading `v`.

For an explicit version:

```bash
uv version 0.1.1
```

For a semantic bump:

```bash
uv version --bump patch
```

Then review and commit the version change:

```bash
git add pyproject.toml uv.lock
git commit -m "Bump version to 0.1.1"
git push origin main
```

Run any relevant tests locally before tagging. A useful release check is:

```bash
rm -rf dist/
uv build --no-sources
uvx twine check dist/*
uv run pytest tests/test_import.py -q
```

## 3. Publish

Create and push an annotated version tag matching `pyproject.toml`:

```bash
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

The workflow accepts stable semantic-version tags such as `v0.1.1` and prerelease tags such as `v0.2.0rc1`, `v0.2.0a1`, and `v0.2.0b1`.

The GitHub Actions workflow then:

1. checks out the tagged source without persisting Git credentials;
2. verifies that the tag matches the version in `pyproject.toml`;
3. builds the wheel and source distribution with `uv build --no-sources`;
4. validates package metadata with `twine check`;
5. installs and imports both the wheel and source distribution in isolated Python 3.12 environments;
6. uploads only the built distributions as a GitHub Actions artifact;
7. runs a separate `publish` job using the protected `pypi` environment;
8. generates PEP 740 attestations;
9. publishes with `uv publish` using PyPI Trusted Publishing/OIDC.

The publish job alone has `id-token: write` permission.

## 4. Post-Release Checks

After the workflow succeeds, verify the release from a clean environment:

```bash
uv run --with jeanspy --no-project -- python -c "import jeanspy; print(jeanspy.__version__)"
```

Also check the PyPI project page to confirm that the README, version, classifiers, project links, wheel, and source distribution look correct.

Optionally create a GitHub Release from the same tag if you want release notes to be visible on GitHub.

## 5. If A Release Fails

Do not reuse a different package build under a version that has already been published to PyPI; PyPI release files are immutable.

If the workflow fails **before** publishing, fix the problem, delete/recreate the local and remote tag if appropriate, and rerun the release from the corrected commit.

If some artifacts were already uploaded to PyPI, inspect the PyPI release first. If a correction requires different package contents, bump to a new version and publish that version instead.

For Trusted Publishing failures, confirm that these values match exactly on both sides:

- GitHub repository: `gomeshun/jeanspy`
- workflow: `.github/workflows/release.yml`
- GitHub environment: `pypi`
- PyPI project: `jeanspy`

No `PYPI_TOKEN`, `UV_PUBLISH_TOKEN`, username, or password should be necessary for the GitHub Actions release workflow.
