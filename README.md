# JeansPy

JeansPy is a Python toolkit for Jeans analysis of dwarf spheroidal galaxies. It combines classical dynamical modeling utilities with optional JAX and NumPyro inference workflows for research use.

## Highlights

- Velocity-dispersion and mass-model calculations based on the Jeans equations
- Optional JAX and NumPyro workflows for gradient-based inference
- ArviZ-compatible posterior storage using `zarr`, `h5netcdf`, or `netCDF4`
- A standard `src` layout suitable for library use, scripts, and notebooks

## Installation

JeansPy requires Python 3.12 or newer.

Install the base package from PyPI:

```bash
pip install jeanspy
```

Install the optional NumPyro and JAX stack for CPU-only environments:

```bash
pip install "jeanspy[numpyro_cpu]"
```

Install the optional NumPyro and JAX stack for CUDA12-backed environments:

```bash
pip install "jeanspy[numpyro_cuda12]"
```

The base install contains the non-JAX runtime. The optional extras add the NumPyro and JAX stack together with the ArviZ storage dependencies used by `jeanspy.sampler_numpyro`.

## Installation From Source

For development with `uv`:

```bash
uv sync
uv sync --extra numpyro_cpu
uv sync --extra numpyro_cuda12
uv sync --extra numpyro_cpu --extra dev
uv sync --extra numpyro_cuda12 --extra dev
```

If you prefer `pip` from a checkout:

```bash
pip install -e .
pip install -e ".[numpyro_cpu]"
pip install -e ".[numpyro_cuda12]"
pip install -e ".[numpyro_cpu,dev]"
pip install -e ".[numpyro_cuda12,dev]"
```

`requirements.txt` keeps the default CUDA12-oriented development environment used in this repository:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import jeanspy as jpy

mdl = jpy.model.get_default_estimation_model(
    dsph_type="Classical",
    dsph_name="Sculptor",
    config="priorconfig.csv",
)

sampler = jpy.sampler.Sampler(mdl)
```

## NumPyro And ArviZ Backends

Both `jeanspy[numpyro_cpu]` and `jeanspy[numpyro_cuda12]` install the backend stack needed by `jeanspy.sampler_numpyro.NumPyroSampler`:

- `arviz`
- `zarr`
- `h5netcdf` and `h5py`
- `netCDF4`
- `xarray`

Storage backend guidance:

- `zarr`: good default for iterative NumPyro runs and append-friendly storage
- `h5netcdf`: good single-file choice when you want an HDF5 or NetCDF-style workflow
- `netcdf4`: good when compatibility with external NetCDF tooling matters most

`NumPyroSampler` defaults to `storage_backend="zarr"`, while still allowing `storage_backend="h5netcdf"` or `storage_backend="netcdf4"`.

## JAX Runtime Configuration

After installing either optional NumPyro extra, the implementation in `model_numpyro` keeps only process-wide JAX settings in environment variables before import. Solver-specific numerical controls are explicit method arguments instead. For example:

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

On GPU float32, `model_numpyro` defaults to `n_u=1024` for the kernel solver. If you need tighter agreement with a high-resolution reference, raise `n_u` explicitly on the relevant `sigmalos2()` call.

To reproduce the backend and precision comparison used during development, run:

```bash
python scripts/compare_runtime_modes.py
```

## Project Links

- Source: https://github.com/gomeshun/jeanspy
- Issues: https://github.com/gomeshun/jeanspy/issues
- Release guide: https://github.com/gomeshun/jeanspy/blob/main/RELEASE.md

## Maintainer Notes

Build and validate distributions locally before upload:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

The full release flow, including TestPyPI upload, is documented in https://github.com/gomeshun/jeanspy/blob/main/RELEASE.md.

## License

JeansPy is distributed under the BSD 3-Clause License. See https://github.com/gomeshun/jeanspy/blob/main/LICENSE for details.
