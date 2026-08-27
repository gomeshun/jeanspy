# JeansPy

## ⚠️ Development Status

> [!WARNING]
> **JeansPy is under active development and may not work correctly.** APIs, numerical behavior, and supported workflows may change without notice. Some parts of the code may be incomplete or insufficiently validated, so results should be independently checked before they are used for scientific conclusions.

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

The following example is self-contained and uses only the base JeansPy installation.

```python
import numpy as np
from jeanspy.model import ConstantAnisotropyModel, DSphModel, NFWModel, PlummerModel

model = DSphModel(
    vmem_kms=0.0,
    submodels={
        "StellarModel": PlummerModel(re_pc=200.0),
        "DMModel": NFWModel(
            rs_pc=1000.0,
            rhos_Msunpc3=1.0e-2,
            r_t_pc=10000.0,
        ),
        "AnisotropyModel": ConstantAnisotropyModel(beta_ani=0.0),
    },
)

R_pc = np.array([50.0, 100.0, 300.0])
sigma_los_kms = model.sigmalos_dequad(R_pc)
print(sigma_los_kms)
```

JeansPy does not bundle an external dwarf-galaxy database. Observational data and object-specific priors should be supplied explicitly by downstream analyses.

## Model Backends

JeansPy provides two supported model backends. The module layout is
intentionally stable for v0.1.0: `jeanspy.model` and
`jeanspy.model_numpyro` are distinct public APIs, and neither backend is
deprecated or “legacy”.

| Backend | Use it when | Capabilities and limitations |
| --- | --- | --- |
| `jeanspy.model` | You need the general-purpose or reference implementation from the base install. | Stateful NumPy/SciPy models, adaptive `dequad` integration, the broader stellar-model collection, J-factor utilities, and the `emcee`-based `jeanspy.sampler` workflow. It is not a JAX/JIT or autodiff API. |
| `jeanspy.model_numpyro` | You need JAX arrays, JIT/autodiff, or NumPyro inference. Install `jeanspy[numpyro_cpu]` or `jeanspy[numpyro_cuda12]`. | Functional models for the currently supported Plummer, NFW, Zhao, and anisotropy paths, with kernel and Abel `sigmalos2` solvers and `jeanspy.sampler_numpyro`. It is not a drop-in replacement for every model, J-factor, or fitting utility in `model`. |

Use `model` when broad model coverage and the established stateful API matter
most. Use `model_numpyro` when differentiable or accelerator-backed inference
matters most; shared calculations are covered by cross-backend numerical
regression tests, but backend-specific solver and precision differences are
intentional.

### Shared model concepts

The physical parameter names are aligned where the models overlap:
`re_pc`, `rs_pc`, `rhos_Msunpc3`, `r_t_pc`, `beta_ani`, `beta_0`,
`beta_inf`, `r_a`, `eta`, and `vmem_kms`. The parameter-passing convention is
backend-specific by design:

| Concept | `model` | `model_numpyro` |
| --- | --- | --- |
| Model parameters | Values are supplied at construction and stored in `model.params`; `update()` changes them. | Values are supplied as a `params` mapping to each numerical method so JAX transformations can trace them. |
| Density | `density_2d(R_pc)` and `density_3d(r_pc)` read the stored parameters. | `density_2d(R_pc, re_pc=...)` and `density_3d(r_pc, re_pc=...)` receive parameters explicitly. |
| Enclosed mass | `enclosed_mass(r_pc)` is the common spelling; the existing `enclosure_mass(r_pc)` spelling remains supported. | `enclosed_mass(r_pc, params=..., method=...)` is canonical; `enclosure_mass(...)` is provided as a compatibility spelling. |
| Line-of-sight dispersion | `DSphModel.sigmalos2(...)` uses the adaptive solver (also available as `sigmalos2_dequad(...)`). | `DSphModel.sigmalos2(..., backend="kernel"|"abel", ...)` uses a JAX-friendly fixed-grid solver. |
| Numerical controls | Adaptive integration controls such as `n` and `n_kernel` are method arguments. | JIT, solver selection, grid sizes, and kernel backend are method arguments. |

## Example Notebooks

The canonical examples are:

- [`demo_model_full.ipynb`](notebooks/demo_model_full.ipynb) visualizes the stellar, anisotropy, and mass-model components. It requires the `numpyro_cpu` or `numpyro_cuda12` extra.
- [`sampler_numpyro_demo.ipynb`](notebooks/sampler_numpyro_demo.ipynb) is the recommended starting point for the checkpointed `NumPyroSampler` workflow. It requires the `numpyro_cpu` or `numpyro_cuda12` extra.

Install an optional CPU environment, including the notebook kernel dependency, with:

```bash
uv sync --extra numpyro_cpu --extra dev
```

The benchmark notebook is a support tool rather than a canonical example:
[`benchmark_jeans_codes.ipynb`](notebooks/benchmark_jeans_codes.ipynb) visualizes
artifacts generated by `scripts/benchmark_jeans_codes.py` and additionally
requires the `benchmark` extra. Generate the artifacts before opening it, for
example:

```bash
uv sync --extra numpyro_cpu --extra benchmark
uv run python scripts/benchmark_jeans_codes.py --quick --n-stars 4000 --engines jeanspy --mock-source jeanspy
```

Notebook runtime outputs are intentionally not committed. The sampler writes
its checkpoint and chunk stores below the ignored `notebooks/_demo_outputs/`
directory.

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

Releases are published automatically from version tags by `.github/workflows/release.yml` using `uv` and PyPI Trusted Publishing. No long-lived PyPI API token is required in GitHub.

Before the first release, configure the `pypi` GitHub environment and the matching PyPI Trusted Publisher. Then create and push a version tag matching the version in `pyproject.toml`, for example:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

See [RELEASE.md](RELEASE.md) for the complete setup, validation, publishing, and recovery procedure.

## License

JeansPy is distributed under the BSD 3-Clause License. See https://github.com/gomeshun/jeanspy/blob/main/LICENSE for details.
