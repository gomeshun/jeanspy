# Installation

Use Python 3.12 or 3.13. The base package requires NumPy, SciPy, pandas, emcee
and h5py. Plotting and the JAX/NumPyro stack are optional dependencies.

For the exact code described by this development guide:

```bash
git clone https://github.com/gomeshun/jeanspy.git
cd jeanspy
git checkout 1a0ad4028d26af1df389ebdfdf992285ec50f8bb
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[numpyro_cpu,plotting]'
```

After the documentation changes reach main, use the matching development
checkout to build this site:

```bash
uv sync --locked --extra docs --extra numpyro_cpu --extra plotting --extra dev
uv run --no-sync python scripts/generate_api_docs.py
uv run --no-sync python scripts/generate_comparison_docs.py
uv run --no-sync python scripts/generate_bibliography.py
uv run --no-sync sphinx-build -W --keep-going -b html docs/source docs/_build/html
uv run --no-sync sphinx-build -W --keep-going -b doctest docs/source docs/_build/doctest
uv run --no-sync python scripts/check_docs_links.py docs/_build/html
```

PyPI publication is planned. Once a release is published, `pip install jeanspy`
will install that release. This page's commands use the explicit development
checkout. Inspect `jeanspy.__version__` and the source commit when comparing
release and development behavior. To add CUDA 12 support, install the `numpyro_cuda12`
extra in a separate environment and verify the effective device with
`jax.devices()`. CUDA installation compatibility alone does not establish
that a particular calculation used a GPU.

Set precision and device before importing JAX or the JAX-backed JeansPy modules:

```bash
JEANSPY_JAX_PLATFORM=cpu JEANSPY_JAX_ENABLE_X64=true python your_analysis.py
```

Keep `uv.lock`, the command, source commit and runtime configuration with the
analysis. Restart identity also checks dependency and backend changes; see
[storage](guides/inference.md).
