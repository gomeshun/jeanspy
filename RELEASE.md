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

### Configure versioned GitHub Pages documentation

Select **GitHub Actions** as the Pages source in repository settings. In the
`github-pages` environment, use selected deployment branches and tags with:

- branch rule `main`, for development documentation;
- tag rule `v*`, for documentation of published releases.

Keep the rule types distinct: a branch named `main` does not permit a release
tag that points to a commit on main. The documentation workflow verifies that
the release tag matches the package version before building the site.

Inspect the current rules before publishing:

```bash
gh api repos/gomeshun/jeanspy/environments/github-pages/deployment-branch-policies
```

The first formal documentation deployment requires both rules. A missing tag
rule blocks the release deployment even if the PR documentation build passes.
The [GitHub environment rules](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments)
describe how the triggering branch or tag is matched.

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
7. GitHub Pages uses the Actions source, and its `github-pages` environment
   allows both `main` and the intended release tag;
8. the version-specific documentation build passes and its installation
   commands name the intended package version and source commit.

For a local metadata sanity check, you can also run:

```bash
rm -rf dist/
uv build --no-sources
uvx twine check dist/*
```

The GitHub release validation is stronger than this local check because it installs the built artifacts into clean environments.

### Distribution layout

The importable package lives in `src/jeanspy`; its three Sersic coefficient
tables live in `src/jeanspy/data` and are accessed through `importlib.resources`.
The wheel contains this runtime package and distribution metadata. The source
distribution additionally includes the test fixtures, executable examples,
output-free notebooks under `notebooks/`, executed documentation notebooks
under `docs/source/`, validation reference and support scripts needed by the
tests. `MANIFEST.in` makes these source-test dependencies explicit.

After a clean build, run `python scripts/check_distribution_contents.py dist`.
It checks required files byte-for-byte against the source, including all
runtime modules and data, and rejects generated chains and untracked runtime
artifacts. The documented notebook outputs are intentional source content.
The release workflow also runs this check before installing the artifacts.

The `benchmark` extra retains JamPy 8.x for the existing cylindrical-quadrature
reference. JamPy 9 removed that solver and uses a different public API and
alignment; its spectral benchmark uses a separate pinned environment. Neither
JamPy version is a base/runtime dependency of JeansPy.

## 3. What The Release Gate Validates

The `tests` job calls `./.github/workflows/test.yml` from the **same commit**
as the caller. It does not check out `main` or trust the CI status of an older
commit. `publish.needs` includes both `build` and `tests`, so a failed,
cancelled or skipped dependency prevents publishing. Ordinary PRs run the
artifact build checks here and the standard Test workflow, without duplicating
the full release matrix. Version-tag pushes run the full matrix before PyPI
publication. To run that matrix before tagging, manually dispatch the
`Publish release to PyPI` workflow on the desired branch; manual runs never
publish, even when dispatched on a tag.

The test definitions are shared with ordinary CI:

| Trigger | Dependency resolution | Python | MCMC |
| --- | --- | --- | --- |
| Ordinary push / PR | locked and lowest compatible direct runtime requirements | 3.12, 3.13 | all chain-generating tests and examples skipped |
| Version tag / manual release validation | locked, fresh, and lowest compatible direct runtime requirements | 3.12, 3.13 | `pytest --run-mcmc`, all tests required |

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
Numerical stress checks remain enabled in ordinary CI. Chain-generating emcee
and NumPyro tests share the `mcmc` marker and run only with `--run-mcmc`;
release validation enables this option in both base and NumPyro jobs and runs
the standalone inference/restart examples. Deterministic likelihood, gradient,
configuration, and identity checks remain in ordinary CI.

Documentation push/PR builds validate saved notebook code/output identities and
execute the four lightweight tutorial notebooks in fresh kernels. MCMC notebooks
retain their saved outputs in ordinary builds. Inference examples and MCMC
notebook regeneration run on a published
GitHub release, or a manual Documentation run with `run_mcmc=true`. These
short-chain checks exercise execution and persistence, not scientific calibration.
CI pins uv to 0.12.15, avoiding the latest-version manifest lookup during setup.

The seven documentation notebooks are the editable sources for Quickstart and
the six tutorial chapters. MyST-NB renders them with execution disabled during
Sphinx builds, and the site supplies a matching `.ipynb` download for each page.
To validate or refresh them locally in the locked documentation environment:

```bash
python scripts/run_doc_notebooks.py --check
python scripts/run_doc_notebooks.py
python scripts/run_doc_notebooks.py --include-mcmc --write
```

The last command explicitly runs MCMC and saves cell outputs, package versions
and hashes of the cell code, lockfile and JeansPy source. Inspect the resulting
figures and short-chain diagnostics before committing refreshed outputs.

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

### Publish and verify the matching documentation

After the PyPI workflow succeeds, **publish a GitHub Release from the same
existing tag** with the reviewed release notes. This is a required part of the
release process: pushing a tag publishes the package, while publishing the
GitHub Release triggers the versioned documentation workflow. A draft GitHub
Release does not publish documentation. Mark an alpha, beta or release-candidate
tag as a prerelease; it must not replace the stable documentation alias.

Wait for the Documentation workflow's build and Pages deployment to succeed,
then check the versioned home page, installation commands, API pages and
version selector. For example, for `v0.1.1` verify:

- `https://gomeshun.github.io/jeanspy/v0.1.1/` identifies version `0.1.1`;
- its `build-info.json` records the exact released commit;
- `https://gomeshun.github.io/jeanspy/versions.json` includes that version;
- `stable/` and the root redirect select the highest formal non-prerelease
  version, and earlier published version directories remain unchanged.

For a prerelease, verify its versioned directory and confirm that `stable/`
still identifies the latest formal release, or remains absent if none exists.

The documentation workflow copies each release once and refuses to replace it
with different content. Review the version-specific preview before publishing.
If documentation deployment fails after a successful package upload, repair
that deployment without recreating or re-uploading the PyPI release.

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
