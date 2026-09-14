# Installation

Use Python 3.12 or 3.13. The base package requires NumPy, SciPy, pandas, emcee
and h5py. Plotting and the JAX/NumPyro stack are optional dependencies.

Create and activate a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
```

On Windows, use `.venv\Scripts\Activate.ps1` in PowerShell instead.

````{only} release
Install the package version documented by this site:

```bash
python -m pip install 'jeanspy==@PACKAGE_VERSION@'
```

For the optional CPU inference and plotting examples:

```bash
python -m pip install 'jeanspy[numpyro_cpu,plotting]==@PACKAGE_VERSION@'
```
````

```{only} development
Use the source installation below for development documentation. Installing
an unpinned PyPI release may provide a different API from this checkout.
```

To install from the source reference used by this documentation build:

```bash
git clone https://github.com/gomeshun/jeanspy.git
cd jeanspy
git checkout @SOURCE_REF@
python -m pip install -e '.[numpyro_cpu,plotting]'
```

To build this site from the same source checkout:

```bash
uv sync --locked --extra docs --extra numpyro_cpu --extra plotting --extra dev
uv run --no-sync python scripts/generate_api_docs.py
uv run --no-sync python scripts/generate_comparison_docs.py
uv run --no-sync python scripts/generate_bibliography.py
uv run --no-sync sphinx-build -W --keep-going -b html docs/source docs/_build/html
uv run --no-sync sphinx-build -W --keep-going -b doctest docs/source docs/_build/doctest
uv run --no-sync python scripts/check_docs_links.py docs/_build/html
```

Inspect `jeanspy.__version__` and the source reference when comparing release
and development behavior. For CUDA 12 support, use a separate environment:

````{only} release
```bash
python -m pip install 'jeanspy[numpyro_cuda12,plotting]==@PACKAGE_VERSION@'
```
````

````{only} development
From the same source checkout:

```bash
python -m pip install -e '.[numpyro_cuda12,plotting]'
```
````

Verify the effective device with `jax.devices()`. CUDA installation
compatibility alone does not establish that a calculation used a GPU.

Set precision and device before importing JAX or the JAX-backed JeansPy modules:

```bash
JEANSPY_JAX_PLATFORM=cpu JEANSPY_JAX_ENABLE_X64=true python your_analysis.py
```

Keep `uv.lock`, the command, source commit and runtime configuration with the
analysis. Restart identity also checks dependency and backend changes; see
[storage](guides/inference.md).
