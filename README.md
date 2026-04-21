# JeansPy
JeansPy is a Python library for the Jeans analysis. This library is designed to help researchers and astrophysicists analyze and understand the dynamics of a given system.

## Features
- Calculation of velocity dispersion profiles using the Jeans equations
- NumPyro/JAX-based inference workflows
- Visualization of results with Matplotlib, plus ArviZ for NumPyro workflows

## Installation

Install the core package with `uv`:

```bash
uv sync
```

Install the optional NumPyro/JAX stack when you need the CUDA12-backed JAX models and samplers:

```bash
uv sync --extra numpyro_cuda12
```

Install the full repository environment, including notebooks, tests, and ancillary plotting/debug tooling:

```bash
uv sync --extra numpyro_cuda12 --extra dev
```

This repository treats `pyproject.toml` as the authoritative package definition. The base install contains the non-JAX runtime; the NumPyro/JAX stack and debug tooling are exposed as optional extras.

The current dependency set targets Python 3.12 or newer because the NumPyro/ArviZ `DataTree` workflow used by `sampler_numpyro` is resolved here against Python 3.12+.

If you need to refresh the environment after changing dependencies, rerun the relevant `uv sync` command for the extras you use.

The `numpyro_cuda12` extra now resolves to `numpyro[cuda12]`, which already pulls `jax[cuda12]`. Installing `.[numpyro_cuda12]` therefore assumes a CUDA 12-capable environment.

If you prefer a pip-based install instead of `uv`, use one of the following:

```bash
pip install -e .                  # core
pip install -e .[numpyro_cuda12]        # core + NumPyro/JAX on CUDA12
pip install -e .[numpyro_cuda12,dev]  # full repo environment
```

For a single-command full repository environment from a checkout, `requirements.txt` installs the editable package with both optional extras:

```bash
pip install -r requirements.txt
```

The repository follows a standard `src` layout:

- `src/jeanspy/`: installable Python package
- `tests/`: test suite
- `scripts/`: standalone analysis and plotting scripts
- `notebooks/`: exploratory notebooks
- `examples/`: sample inputs and example notebooks


## ArviZ I/O Backends

This project uses ArviZ 1.x `DataTree` outputs for NumPyro workflows. The `numpyro` extra installs the ArviZ stack together with the backend packages needed by `sampler_numpyro`:

- `arviz`
- `zarr`
- `h5netcdf` and `h5py`
- `netCDF4`
- `xarray`

Install them with `jeanspy[numpyro_cuda12]`, `uv sync --extra numpyro_cuda12`, or the full `requirements.txt` environment.

Backend guidance for `jeanspy.sampler_numpyro.NumPyroSampler`:

- `zarr`: best default for iterative NumPyro runs. It is chunked, append-friendly, and a natural fit for repeated save/load workflows.
- `h5netcdf`: best single-file alternative when you want HDF5/NetCDF storage with a Python-first stack.
- `netcdf4`: best when compatibility with external NetCDF tooling matters most.


`NumPyroSampler` defaults to `storage_backend="zarr"`, while still allowing `storage_backend="h5netcdf"` or `storage_backend="netcdf4"`.

## Build And Publish

Build a source distribution and wheel from the repository root:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

If the generated artifacts are valid, upload them to PyPI:

```bash
python -m twine upload dist/*
```

## Usage

```python
import jeanspy as jpy

# define your model or load the preset model
mdl = jpy.model.get_default_estimation_model(
    dsph_type="Classical",  # "Classical" or "UFD"
    dsph_name="Sculptor",
    config="priorconfig.csv",
)

# run the estimation
sampler = jpy.sampler.Sampler(mdl)
```

## JAX Runtime Configuration

After installing the `numpyro_cuda12` extra, the NumPyro/JAX implementation in [src/jeanspy/model_numpyro.py](src/jeanspy/model_numpyro.py) keeps only process-wide JAX settings in environment variables before import:


The numerical knobs that only affect a specific solver call are now explicit method arguments instead of environment variables. For example:

```python
from jeanspy.model_numpyro import ConstantAnisotropyModel, DSphModel, NFWModel, PlummerModel

dsph = DSphModel(
    submodels={
        "StellarModel": PlummerModel(),
        "DMModel": NFWModel(),
        "AnisotropyModel": ConstantAnisotropyModel(),
    }
)

sigma2 = dsph.sigmalos2(
    R_pc,
    params=params,
    backend="kernel",
    jit=True,
    n_u=1024,
    u_max=5000.0,
    constant_kernel_backend="jax",
    n_kernel=64,
)
```

For direct constant-anisotropy kernel comparisons, choose the backend per call:

```python
kernel = ConstantAnisotropyModel().kernel(
    u,
    R_pc,
    params={"beta_ani": 0.5},
    backend="scipy",
)
```

On GPU float32, `model_numpyro` defaults to `n_u=1024` for the kernel solver. That keeps the grid much smaller than the previous development setting while still staying in the sub-`1e-3` relative-error range on the representative constant-anisotropy benchmark cases used during development. If you need tighter agreement with the high-resolution reference, raise `n_u` explicitly on the relevant `sigmalos2()` call.

To reproduce the backend and precision comparison used during development, run:

```bash
python scripts/compare_runtime_modes.py
```

## License

JeansPy is distributed under the terms of the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
