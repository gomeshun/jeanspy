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

JeansPy supports CPython 3.12 and 3.13. The base install contains the
numerical modeling runtime and the emcee sampler; optional JAX/NumPyro and
plotting features are kept out of the base install.

Install the base package from PyPI:

```bash
pip install jeanspy
```

Install the plotting helpers and the `jfactor` command-line plotting support:

```bash
pip install "jeanspy[plotting]"
```

Install the optional NumPyro and JAX stack for CPU-only environments:

```bash
pip install "jeanspy[numpyro_cpu]"
```

Install the optional NumPyro and JAX stack for CUDA12-backed environments:

```bash
pip install "jeanspy[numpyro_cuda12]"
```

The base dependencies are NumPy, pandas, SciPy, emcee, and h5py. The
`jeanspy.sampler.Sampler` API is a supported emcee-based inference feature;
plotting imports are only needed for plotting helpers or the `jfactor`
command-line demonstration. The NumPyro extras add JAX, NumPyro, and the
ArviZ storage dependencies used by `jeanspy.sampler_numpyro`.

### Supported environment matrix

The v0.1.0 support matrix is:

| Install | Python | JAX | NumPyro | Accelerator |
| --- | --- | --- | --- | --- |
| `jeanspy` | 3.12, 3.13 | — | — | CPU |
| `jeanspy[numpyro_cpu]` | 3.12, 3.13 | `jax[cpu] >=0.4.35` | `numpyro[cpu] >=0.18.0` | CPU |
| `jeanspy[numpyro_cuda12]` | 3.12, 3.13 | `jax[cuda12] >=0.7.0` | `numpyro[cuda12] >=0.20.0` | CUDA 12 |

The NumPyro extras also require ArviZ 1.0, xarray 2024.11 or newer, and their
declared storage backends. ArviZ 1.0 requires Python 3.12 and NumPy 2 or
newer, which is why Python 3.11 is not in this release's matrix. The
dependency ranges intentionally use API-compatible lower bounds rather than
the versions in one development environment. CI resolves and tests the
latest versions satisfying these ranges on both supported Python versions;
the CUDA extra is installation-compatible but is not run on the CPU-only CI
runner.

## Installation From Source

For development with `uv`:

```bash
uv sync
uv sync --extra plotting
uv sync --extra numpyro_cpu
uv sync --extra numpyro_cuda12
uv sync --extra numpyro_cpu --extra dev --extra plotting
uv sync --extra numpyro_cuda12 --extra dev --extra plotting
```

If you prefer `pip` from a checkout:

```bash
pip install -e .
pip install -e ".[plotting]"
pip install -e ".[numpyro_cpu]"
pip install -e ".[numpyro_cuda12]"
pip install -e ".[numpyro_cpu,dev,plotting]"
pip install -e ".[numpyro_cuda12,dev,plotting]"
```

`requirements.txt` is a CPU-first full development environment. It does not
install CUDA packages; use the `numpyro_cuda12` extra explicitly when CUDA 12
is available:

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
