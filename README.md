# JeansPy

## ⚠️ Development Status

> [!WARNING]
> **JeansPy is under active development and may not work correctly.** APIs, numerical behavior, and supported workflows may change without notice. Some parts of the code may be incomplete or insufficiently validated, so results should be independently checked before they are used for scientific conclusions.

JeansPy is a Python toolkit for Jeans analysis of dwarf spheroidal galaxies. It combines classical dynamical modeling utilities with optional JAX and NumPyro inference workflows for research use.

## Highlights

- Velocity-dispersion and mass-model calculations based on the Jeans equations
- [Axisymmetric Jeans forward models](docs/axisymmetric.md) following Hayashi & Chiba, with flattened tracers/halos and inclination-dependent LOS moments
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

Install the optional plotting helpers:

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
plotting imports are only needed for plotting helpers. The NumPyro extras add
JAX, NumPyro, and the ArviZ storage dependencies used by
`jeanspy.sampler_numpyro`.

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
the versions in one development environment. Ordinary CI tests `uv.lock`;
release-related PRs and tags additionally resolve and test the latest versions
satisfying these ranges, on both supported Python versions, including opt-in
MCMC tests. All of these tests and the built-artifact checks must pass before
publication. The exact resolved versions are retained as CI artifacts;
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

### Classical inference with explicit priors

`get_default_estimation_model(data, photometry_prior_loc,
photometry_prior_scale, config=...)` composes Plummer + NFW + constant
anisotropy. Supply kinematic columns `R_pc`, `vlos_kms`, and `e_vlos_kms`
and a DataFrame or CSV with finite `lower < upper` bounds in this order:

| Sampling coordinate | Physical parameter |
| --- | --- |
| `vmem_kms` | systemic velocity in km/s |
| `log10_re_pc` | `re_pc = 10**log10_re_pc` |
| `log10_rs_pc` | `rs_pc = 10**log10_rs_pc` |
| `log10_rhos_Msunpc3` | `rhos_Msunpc3 = 10**log10_rhos_Msunpc3` |
| `log10_r_t_pc` | `r_t_pc = 10**log10_r_t_pc` |
| `bfunc_beta_ani` | `beta_ani = 1 - 10**bfunc_beta_ani` |

The photometry prior is Gaussian in `log10_re_pc`; its location and scale
are explicitly supplied, with a positive finite scale. A uniform prior in
these sampling coordinates is **not** uniform in physical radius/density or
anisotropy. Choose bounds appropriate to your scientific analysis.

A missing config path produces an **unfilled template and raises ValueError**.
Fill it in before retrying; the library does not invent universal prior ranges.
Duplicate, missing, misordered names and nonfinite/degenerate bounds fail
before sampling. `model.sample(size)` draws within the finite support,
including the truncated photometry prior, and can be passed directly as the
`p0_generator` to `jeanspy.sampler.Sampler`.

`reset_data()` preserves supplied priors and invalidates cached WBIC
temperature. The explicit `vmem_prior_from_data=True` option replaces only
the velocity bounds by the data minimum/maximum at each reset; this is an
empirical-prior choice and requires a nonzero velocity range. Sampling and
prior evaluation read the same updated bounds. Shared data can only be
updated at its original length while workers are idle.
WBIC requires at least two observations; an ordinary likelihood can use one.

`Sampler` checks the saved analysis identity before restarting or appending.
Changing observations or priors requires a new output prefix, or an explicit
`reset=True` to discard that backend's chain. `burn_in()` continues from the
final warmup ensemble; its draws remain stored for compatibility. Exclude them
with `get_chain(discard=n_warmup)` and the corresponding log-probability methods.

Run the reproducible synthetic example, including persisted-chain restart:

```bash
python scripts/example_classical_inference.py --output-dir /tmp/jeanspy-example
```

Use a new output directory. The script records the synthetic observations,
explicit example priors, chain and verification result. Its short chain
tests the workflow; it does not establish convergence or scientific coverage.

### J-factor calculations

The supported J-factor API is provided by the classical dark-matter model
methods. The historical `jeanspy.jfactor` module and its plotting command are
not shipped in v0.1.0. Replace standalone calls with
`DMModel.jfactor_ullio2016` or `DMModel.jfactor_ullio2016_simple`:

```python
from jeanspy.model import NFWModel

dm = NFWModel(
    rs_pc=1000.0,
    rhos_Msunpc3=1.0e-2,
    r_t_pc=10000.0,
)

j_full = dm.jfactor_ullio2016(dist_pc=30000.0, roi_deg=0.5)
j_spherical = dm.jfactor_ullio2016_simple(dist_pc=30000.0, roi_deg=0.5)
```

Annihilation luminosity imposes a stricter central-cusp condition than mass:
Zhao J-factors require **`g < 1.5`**, even though enclosed mass exists for
`g < 3`. Divergent cusps and failed quadratures raise `ValueError`; no implicit
central cutoff is introduced. NFW/Zhao profile scales must be finite and
positive. The Evans NFW method uses a stable series around `R/rs=1`. It retains
the historical infinite-line-of-sight approximation with the projected
aperture capped at `r_t_pc`; this is not a three-dimensional truncation.
Use `jfactor_ullio2016` for that geometry.

`StellarModel.density_2d_truncated(R, R_trunc)` vanishes outside the cutoff and
integrates to one over the whole plane. `Uniform2dModel` likewise has support
only inside its disk, and its radial CDF saturates at one outside the disk.

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

Zhao mass supports `a > 0`, `g < 3`, finite `b` (including `b <= 3`),
positive finite scale radius/density, and positive truncation radius. The
requested radius must be nonnegative and finite after truncation. Its numerical
integral removes the central cusp with a power substitution and integrates the
outer profile in log radius, without a central cutoff. Both classical mass
spellings and JAX `auto`/`numeric` mass accept `n_steps=128` (Gauss nodes per
segment); JAX Jeans methods expose this as `dm_mass_n_steps`. Increase this
independently of `n_u`/`n_r` to check mass and LOS convergence separately.
The explicit JAX analytic path falls back to this integral at `b <= 3` or
saturated beta arguments; the exact NFW limit still uses its stable closed form.

Regression tests compare independent SciPy integration on a grid spanning
`a=0.5..5`, `b=2..8`, `g=0..2.99`, and truncated `r/rs=1e-6..1e6`:
relative tolerances are `1e-6` in float64 and `5e-5` in float32 for default
numerical mass. These are tested grid bounds, not an accuracy guarantee for
all Zhao parameters or for the outer Jeans integral. Check convergence outside
that grid, especially closer to `g=3`. Classical invalid mass inputs raise
`ValueError`; JAX returns NaN under eager execution and JIT. Jeans solvers
preserve invalid mass signals, and the NumPyro likelihood rejects invalid
velocity variances with negative infinite log probability.

### Projected-radius domain

All LOS solvers accept a scalar or a **nonempty one-dimensional** array of
projected radii. Only **finite `R_pc > 0`** are supported. `R=0` needs a
model-dependent central-limit calculation, which these solvers do not provide;
it must not be approximated by an arbitrary epsilon.

Classical solvers raise `ValueError` if any radius is invalid and preserve
scalar versus vector output. JAX solvers always return a one-dimensional
array (length one for a scalar). Invalid radius elements return NaN in both
eager and JIT execution; valid members of a mixed array are unaffected.
Invalid shapes raise `ValueError` in both backends. The NumPyro likelihood
rejects invalid solver results with negative infinite log probability.

`JeansLikelihoodModel` requires three matching, nonempty one-dimensional
observation arrays. It never broadcasts a velocity column of shape `(N, 1)`
against radii of shape `(N,)`. All observations must be finite, radii positive,
and measurement errors nonnegative. Invalid dynamic values are rejected with
negative infinite log probability, including under JIT; shape errors raise
`ValueError`. A custom velocity mean must be scalar or have shape `(N,)`.

Small positive radii can require much larger `u_max`: kernel radii extend only
to `R_pc*u_max`. The default accuracy envelope starts at `R/Re=0.005`.
The near-center regression additionally compares isotropic Plummer + NFW
against an independent adaptive integral at `R=0.001..300 pc` for `Re=200 pc`,
with `u_max=1e8`, `n_u=1025` (kernel) or `n_r=8192` (Abel), then refines these
controls. These are convergence-test settings, not new universal defaults.
The Abel method uses a piecewise radial grid: its radius derivatives are
defined away from bin-boundary crossings, where the approximation has kinks.

Numerical convergence does not establish a nonnegative stellar distribution
function or calibrated statistical intervals. For a finite central potential,
the central cusp/anisotropy condition is `gamma_star >= 2*beta_0`
([An & Evans 2006](https://arxiv.org/abs/astro-ph/0511686)). In particular,
a cored Plummer tracer with an NFW halo and constant `beta_ani > 0` violates
this necessary condition. Radial-anisotropy stress tests check the formal
Jeans integrals, not the physical admissibility of those models. Choose priors
and perform distribution-function and coverage checks for the scientific use.

## Example Notebooks

The recommended starting point is:

- [`demo_model_full.ipynb`](notebooks/demo_model_full.ipynb): a top-to-bottom Getting Started tutorial for the classical API, including stellar/DM/anisotropy components, `DSphModel`, line-of-sight velocity dispersion, J-factors, and Sérsic deprojection. It requires the `plotting` extra.

For gradient-based inference and checkpointed NumPyro sampling, continue with:

- [`sampler_numpyro_demo.ipynb`](notebooks/sampler_numpyro_demo.ipynb): builds a reusable Jeans likelihood, runs NUTS, resumes from a checkpoint, and combines ArviZ outputs. It requires the `numpyro_cpu` or `numpyro_cuda12` extra.

From a repository checkout, install the corresponding notebook environment with for example:

```bash
uv sync --extra plotting --extra dev
uv sync --extra numpyro_cpu --extra dev --extra plotting
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
directory. The canonical Getting Started notebook is also executed cell-by-cell in CI so API changes cannot silently leave the public example broken.

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

### Safe restart contract

Both samplers fingerprint the model, priors, observation contents, parameter
schema, numerical configuration, package source, and relevant library versions.
NumPyro additionally records the model's call arguments, chain configuration,
active CPU/GPU backend, and JAX precision settings, since solver defaults can
depend on the backend.
Repeated runs and checkpoints must match this identity before any samples are
appended or cached energies reused. A changed target requires a **new
`output_dir`**, even with `resume=False`; also construct a new `MCMC` instance.
`load_checkpoint()` verifies the target immediately and `run()` verifies the
observations before using the loaded state. Raw MCMC state produced outside
the wrapper is not adopted automatically.

Pre-identity checkpoints and emcee backends remain readable as historical
results but cannot safely resume. Preserve them and start a new output location;
do not copy metadata from another analysis to bypass the check. A dependency or
source upgrade conservatively requires a new chain as well.

Ordinary NumPyro distributions, parameter specs, arrays, Python closures, and
the supplied Jeans models are supported automatically. Custom models with
hidden state (for example file contents, remote data, or opaque extension
objects) must expose a `sampling_identity()` method returning a deterministic
mapping of all target-defining state. This user-supplied contract must change
whenever that state changes. The guard cannot infer arbitrary Python side
effects. Treat low-level manual sample stores as user-supplied data.

`ParameterSpec.exp` and `.pow10` accept an omitted `param_name`: the transformed
value then keeps `sample_name` as its parameter-dictionary key and is recorded
at `sample_name + "_transformed"`. Supply `param_name` to map it to a physical
Jeans parameter. Duplicate sample, observation, or physical parameter names
are rejected.

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
