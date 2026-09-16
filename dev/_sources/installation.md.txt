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

## Run the notebooks

The [Quickstart](quickstart.ipynb) and [tutorials](tutorials/index.md) are
Jupyter notebooks with saved outputs. Each page has a **Download this notebook**
link. Install a notebook frontend in the same environment as JeansPy:

```bash
python -m pip install jupyterlab
jupyter lab
```

Open the downloaded `.ipynb`, select the environment containing JeansPy, and
use **Restart Kernel and Run All Cells**. Each notebook contains its own setup
and synthetic inputs, so it can run without a source checkout or another
notebook's state. The website displays saved outputs; running Python cells
requires a local Jupyter kernel.
