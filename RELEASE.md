# Releasing jeanspy to PyPI

JeansPy is published from GitHub Actions with `uv` and PyPI Trusted Publishing. The release workflow is `.github/workflows/release.yml`.

No long-lived PyPI API token is stored in GitHub. A version tag triggers both
artifact validation and the full release test matrix used on release-related
pull requests. The publishing job depends on **both** jobs succeeding; a
build/smoke pass alone cannot publish a commit with failing tests. The exact
validated wheel and source distribution are passed to the publishing job
with GitHub OIDC permission.

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

### TestPyPI policy

TestPyPI is an **optional manual preflight**, not part of the automated release gate. The workflow validates the exact wheel and sdist locally before publishing, while TestPyPI requires a second publisher configuration and can give misleading dependency-install results unless PyPI is also configured as an extra index.

For the first public release, using TestPyPI once is recommended if you want to inspect the rendered project page and upload metadata before the real upload. It is not required for subsequent releases unless the packaging or publishing setup changes.

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

Before tagging, all of the following should be true:

1. ordinary push/PR CI is green;
2. the release workflow's `Build and validate distributions` and `Release test matrix` jobs are green on the release-related PR;
3. `pyproject.toml` contains the intended version;
4. `README.md` contains the Quick Start that should be executable by a base installation;
5. the `pypi` GitHub environment and PyPI Trusted Publisher still match the repository/workflow configuration;
6. any intentional packaging changes, optional-dependency changes, or package-data changes have been reviewed.

For a local metadata sanity check, you can also run:

```bash
rm -rf dist/
uv build --no-sources
uvx twine check dist/*
```

The GitHub release validation is stronger than this local check because it installs the built artifacts into clean environments.

## 3. What The Release Gate Validates

The `tests` job calls `./.github/workflows/test.yml` from the **same commit**
as the caller. It does not check out `main` or trust the CI status of an older
commit. `publish.needs` includes both `build` and `tests`, so a failed,
cancelled or skipped dependency prevents publishing. No tag is needed to
exercise validation: PRs changing either workflow, `pyproject.toml`, `uv.lock`,
the README, this guide or the artifact validator run the release gate with
publishing skipped.

The test definitions are shared with ordinary CI:

| Trigger | Dependency resolution | Python | MCMC |
| --- | --- | --- | --- |
| Ordinary push / PR | locked and lowest compatible direct runtime requirements | 3.12, 3.13 | opt-in tests skipped |
| Release-related PR / version tag | locked, fresh, and lowest compatible direct runtime requirements | 3.12, 3.13 | `pytest --run-mcmc`, all tests required |

Base numerical/inference and optional-dependency isolation checks run on Linux,
Windows, and macOS (locked and lowest, and additionally fresh for release validation).
The complete NumPyro CPU suite runs on Linux in all listed resolution modes.
The lowest mode resolves runtime/plotting requirements with
`uv pip compile --resolution lowest-direct --only-binary :all:` for each Python
version, then adds the test tools. It checks actual imports, inference, and
storage, because dependency metadata alone cannot detect a NumPy binary ABI
mismatch. Fresh installations use `uv pip install` into a new venv;
subsequent commands use that interpreter directly so `uv run` cannot silently
restore locked dependencies. Resolved versions are uploaded for auditing.
The existing numerical stress checks and examples run in the same matrix.
CUDA extras are resolution-checked below; these runners do not validate GPU
execution or GPU performance. Record a separate GPU smoke result when changing
JAX numerical or sampler code.

The binary-compatibility floors are pandas 2.2.2, h5py 3.11.0, and netCDF4 1.7.4.
netCDF4 1.7.2 passed an isolated import but failed to write after h5py 3.11.0
had been imported on Linux; 1.7.4 passed both import orders and read/write
checks. Dedicated subprocess tests cover both orders. The storage floors are
xarray 2025.3.1 and Zarr 3.0.8, whose DataTree/Zarr APIs passed the actual
three-backend sampler tests; xarray 2024.11.0 failed with Zarr 3.
These permit NumPy 1.x/2.x where the complete dependency set allows it; the CPU
ArviZ stack itself currently requires NumPy 2 or newer. See the
[pandas 2.2.2 release notes](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v2.2.2.html)
for its first generally NumPy-2-compatible wheels. The actual lowest versions
vary with Python and wheel availability and are recorded by each job.

The ordinary/release tests and artifact validation are read-only with respect
to PyPI. Only the tag-only publish job has `id-token: write` and enters the
`pypi` environment. Never push a release tag merely to test the workflow.

For pull requests that change release-related files, and again for a release tag, `.github/workflows/release.yml`:

1. builds both the wheel and source distribution with `uv build --no-sources`;
2. validates package metadata with `twine check`;
3. resolves `numpyro_cpu` and `numpyro_cuda12` from both built artifacts with `uv pip install --dry-run`;
4. installs the wheel into a fresh Python 3.12 environment;
5. executes the Python block in the README `Quick Start` against that installed wheel;
6. verifies packaged runtime data files are present and can be consumed by `SersicModel`;
7. repeats the base Quick Start and package-data checks from a freshly installed sdist;
8. installs the wheel with the `numpyro_cpu` extra in another fresh environment;
9. forces the JAX CPU backend and runs a NumPyro/Jeans likelihood smoke test, including `sigmalos2` and a traced `JeansLikelihoodModel`;
10. uploads only the validated wheel and sdist as the `dist` artifact.

The validation script rejects imports that come from the repository source checkout, so the release gate cannot accidentally pass by testing `src/jeanspy` instead of the installed distribution.

The `Publish to PyPI` job runs only for matching version tags and depends on the validation job. It downloads the same validated `dist` artifact, generates PEP 740 attestations, and publishes with `uv publish`. The publish job alone has `id-token: write` permission.

## 4. Publish

Create and push an annotated version tag matching `pyproject.toml`:

```bash
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

The workflow accepts stable semantic-version tags such as `v0.1.1` and prerelease tags such as `v0.2.0rc1`, `v0.2.0a1`, and `v0.2.0b1`.

On a tag, the workflow first verifies that the tag matches the version in `pyproject.toml`. A mismatch stops the release before artifacts are published.

Do not manually rebuild artifacts between validation and publishing. The publishing job intentionally consumes the exact wheel and sdist produced by the successful validation job.

## 5. Post-Release Checks

After the workflow succeeds, verify the release from a clean environment:

```bash
uv run --with jeanspy --no-project -- python -c "import jeanspy; print(jeanspy.__version__)"
```

Also check the PyPI project page to confirm that the README, version, classifiers, project links, wheel, and source distribution look correct.

For releases that change NumPyro packaging, it is also useful to verify the CPU extra from PyPI:

```bash
uv run --with "jeanspy[numpyro_cpu]" --no-project -- python -c "import jax, jeanspy; print(jeanspy.__version__, jax.default_backend())"
```

Optionally create a GitHub Release from the same tag if you want release notes to be visible on GitHub.

## 6. If A Release Fails

Do not reuse a different package build under a version that has already been published to PyPI; PyPI release files are immutable.

### Validation fails before publishing

No package has been uploaded. Fix the underlying code, metadata, README Quick Start, package data, or dependency specification on a normal branch/PR. Get CI and the release validation green before tagging again.

If a bad tag was pushed and **nothing was published**, delete and recreate the tag only after the corrected commit is ready:

```bash
git tag -d v0.1.1
git push --delete origin v0.1.1
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

### Trusted Publishing or publish step fails before any file is accepted

Confirm that these values match exactly on both sides:

- GitHub repository: `gomeshun/jeanspy`
- workflow: `.github/workflows/release.yml`
- GitHub environment: `pypi`
- PyPI project: `jeanspy`

Also confirm that the publish job reached the protected `pypi` environment and still has `id-token: write`.

If PyPI accepted no files, correct the publishing configuration and rerun from the same validated commit/tag as appropriate.

### PyPI accepted one or more files

Treat the version as used. Do not replace or overwrite those artifacts. Inspect the PyPI release, fix the problem in the repository, bump to a new version, pass the release gate again, and publish the new version.

No `PYPI_TOKEN`, `UV_PUBLISH_TOKEN`, username, or password should be necessary for the GitHub Actions release workflow.
