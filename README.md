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

## Model Backends

JeansPy provides two supported model backends. The module layout is
intentionally stable for v0.1.0: `jeanspy.model` and
`jeanspy.model_numpyro` are distinct public APIs, and neither backend is
deprecated or “legacy”.

| Backend | Use it when | Capabilities and limitations |
| --- | --- | --- |
| `jeanspy.model` | You need the general-purpose or reference implementation from the base install. | Stateful NumPy/SciPy models, fixed-grid double-exponential (`dequad`) integration, the broader stellar-model collection, J-factor utilities, and the `emcee`-based `jeanspy.sampler` workflow. It is not a JAX/JIT or autodiff API. |
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
| Line-of-sight dispersion | `DSphModel.sigmalos2(...)` uses the classical fixed-grid double-exponential (`dequad`) solver (also available as `sigmalos2_dequad(...)`). | `DSphModel.sigmalos2(..., backend="kernel"|"abel", ...)` uses a JAX-friendly fixed-grid solver. |
| Numerical controls | Quadrature/grid resolution controls such as `n` and `n_kernel` are method arguments. | JIT, solver selection, grid sizes, and kernel backend are method arguments. |

For `model_numpyro`, `DMModel.enclosed_mass(..., method="auto")` is the
default model-aware choice: it uses the analytic NFW mass and the fixed-grid
numeric mass for Zhao. The Zhao analytic mass uses
`jax.scipy.special.betainc`; JAX does not provide autodiff through its shape
parameters, so gradients through Zhao `a`, `b`, or `g` can fail on that path.
The default `DSphModel.sigmalos2(..., dm_mass_method="auto")` follows the same
choice and is the NUTS-safe path. Use `dm_mass_method="numeric"` to force
numeric mass, and request `method="analytic"` or
`dm_mass_method="analytic"` explicitly only when the closed form is desired
without those Zhao shape-parameter gradients.

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

### Kernel `sigmalos2` numerical-accuracy contract

For calls resolved to `backend="kernel"` with the default `sqrtlog` outer
transform, the maintained numerical target is a maximum relative error of
`1e-3` against a high-resolution float64 kernel reference.  The regression
metric uses a floor of `max(1e-12, 1e-9 * max(abs(reference)))` so values that
are numerically negligible do not dominate the relative-error statistic.

The deterministic stress benchmark samples the following dSph-oriented
envelope with a Plummer tracer and NFW halo:

- `0.005 <= R/Re <= 10`;
- `0.05 <= rs/Re <= 100` and the standard benchmark truncation `r_t/Re = 40`;
- constant anisotropy from `beta=-9` through `beta=0.98`;
- Osipkov-Merritt transitions with `0.005 <= r_a/Re <= 50`;
- Baes-van Hese models with `0.1 <= eta <= 10`, anisotropy edges down to
  `beta=-9` and up to `beta=0.98`, and `0.005 <= r_a/Re <= 50`.

This is a tested numerical envelope, not a proof for every continuous point in
that box or for arbitrary tracer/halo profiles.  Zhao halos, more extreme
anisotropy, `eta > 10`, substantially more extended tails, or radii outside the
sampled range should be convergence-tested explicitly.

The kernel defaults are tuned to the tail-dominated error found in the stress
study: CPU float64/float32 use `n_u=128`, GPU-oriented float32 keeps
`n_u=1024`, and both use `u_max=10000`.  The Baes inner quadrature remains
`n_kernel=32`; constant-anisotropy JAX kernels use 32 nodes on CPU and 64 on
GPU float32.  Increasing `n_u` at fixed, too-small `u_max` does not repair tail
truncation, so for extended models increase `u_max` first and then increase
`n_u` if the denser interval still changes the result appreciably.

A practical convergence check is to recompute the result after increasing
`u_max` and then doubling `n_u`.  For generic Baes models, increase `n_kernel`
only if the inner-kernel quadrature itself is suspected.  The Abel solver is a
useful independent cross-check, but it has a separate radial discretization
controlled by `n_r`; the `1e-3` contract above does not automatically apply to
an `auto` call that resolves to the Abel backend.

Run the compact CI regression set with:

```bash
JAX_ENABLE_X64=true python scripts/benchmark_sigmalos2_accuracy_contract.py
```

Run the complete sampled prior-edge matrix with:

```bash
JAX_ENABLE_X64=true python scripts/benchmark_sigmalos2_accuracy_contract.py --full
```

GitHub-hosted CI has no GPU.  The benchmark therefore evaluates the
GPU-oriented float32 numerical grid on CPU as an arithmetic/accuracy proxy;
GPU wall-clock performance must be measured on actual accelerator hardware.

To reproduce the broader backend and precision comparison used during development, run:

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
