# Jeans Code Benchmark

This directory contains a reproducible benchmark harness for comparing `jeanspy`
against external Jeans-analysis codes under a common profile-likelihood setup.

The current runs use Plummer tracer plus NFW dark-matter mock data generated
through the `external/create_mock` submodule. The Plummer projected half-light
radius is fixed to `re_pc = 200 pc`. The fitted parameters use log-flat priors
in `rs_pc`, `rhos_Msunpc3`, and, for Osipkov-Merritt runs, `r_a_pc`.

## Implemented Targets

| Target | Status | Notes |
|---|---:|---|
| `jeanspy` / emcee | included | Uses `jeanspy.model_numpyro.DSphModel.sigmalos2` with the common emcee wrapper. |
| `jeanspy-nuts` | included | Uses the same `DSphModel.sigmalos2` likelihood through NumPyro NUTS. |
| `jampy` | included | Uses `jampy.sph.proj`; Plummer and NFW are pre-fit as MGEs and MGE setup is excluded from sampling time. |
| `AGAMA` | included | Uses `agama.DistributionFunction(type="QuasiSpherical")` and `GalaxyModel.moments`; Bayesian sampling is external. |
| `gravsphere-v1` | included | Uses the official legacy `justinread/gravsphere` `functions.sigp` second-moment routine through a compatibility adapter and the common emcee wrapper. |
| `GravSphere2` | skipped | LOS-only mode still calls `sigp_k` and `lnpdfj`, so the native workflow fits a second+fourth-moment velocity PDF rather than a Gaussian second-moment-only profile likelihood. |
| `pyGravSphere` | skipped | The SWIG/C API exposes fast binned second-moment calls, but the published wrapper targets legacy Python/SWIG builds and was not installable in this Python 3.12 uv environment without a port. |

AGAMA is much slower per likelihood evaluation in this wrapper because the
QuasiSpherical DF is rebuilt for each parameter proposal. The CLI skips engines
whose median prediction exceeds `--slow-eval-threshold` unless `--include-slow`
is set.

## GravSphere Notes

The re-check changed the GravSphere handling:

- `dadams42/GravSphere2` can be run without proper motions, but that is not a
  second-moment-only mode. Its LOS likelihood computes projected dispersion and
  kurtosis with `sigp_k`, then evaluates a non-Gaussian velocity PDF with
  `lnpdfj`.
- `justinread/gravsphere` is the better provenance target for the classic
  second-moment benchmark. The full runner is a legacy script stack, so the
  benchmark calls the official low-level `functions.sigp` routine directly.
- `AnnaGenina/pyGravSphere` may be the faster backend path, but it needs a
  Python 3/SWIG rebuild before it can be a reproducible uv dependency here.

## Quick Results

These are short smoke-test chains, not production convergence runs.

### Code Comparison With GravSphere v1

Commands:

```bash
uv run python scripts/benchmark_jeans_codes.py --anisotropy isotropic --n-stars 4000 --n-bins 12 --engines jeanspy,jampy,gravsphere-v1,gravsphere2,pygravsphere --include-slow --seed 62345 --n-walkers 18 --n-steps 120 --n-burn 40 --jeanspy-n-u 96 --report-path benchmarks/jeans_comparison/README_gravsphere_v1_isotropic.md

uv run python scripts/benchmark_jeans_codes.py --anisotropy om --n-stars 4000 --n-bins 12 --engines jeanspy,jampy,gravsphere-v1,gravsphere2,pygravsphere --include-slow --seed 62346 --n-walkers 18 --n-steps 120 --n-burn 40 --jeanspy-n-u 96 --report-path benchmarks/jeans_comparison/README_gravsphere_v1_om.md
```

| Anisotropy | Engine | Sampling s | Median predict s | Acceptance | q50 log10_rs_pc | q50 log10_rhos | q50 log10_r_a_pc |
|---|---|---:|---:|---:|---:|---:|---:|
| isotropic | `jeanspy` | 1.18 | 0.000568 | 0.677 | 3.0156 | -1.8566 |  |
| isotropic | `jampy` | 5.87 | 0.00269 | 0.684 | 3.0015 | -1.8379 |  |
| isotropic | `gravsphere-v1` | 57.9 | 0.0262 | 0.708 | 3.0003 | -1.8369 |  |
| OM | `jeanspy` | 1.19 | 0.000569 | 0.581 | 3.2970 | -2.2108 | 2.6587 |
| OM | `jampy` | 6.11 | 0.00278 | 0.587 | 3.2648 | -2.1708 | 2.6474 |
| OM | `gravsphere-v1` | 58.7 | 0.0267 | 0.569 | 3.3153 | -2.2374 | 2.6196 |

The legacy GravSphere v1 adapter gives posterior medians consistent with
`jeanspy` and `jampy` under the same likelihood, but is roughly 48-52x slower
than `jeanspy` for these short emcee runs.

### JeansPy Sampler Comparison

Commands:

```bash
uv run python scripts/benchmark_jeans_codes.py --anisotropy isotropic --n-stars 4000 --n-bins 12 --engines jeanspy,jeanspy-nuts --seed 52345 --n-walkers 18 --n-steps 120 --n-burn 40 --nuts-warmup 200 --nuts-samples 400 --jeanspy-n-u 96 --report-path benchmarks/jeans_comparison/README_sampler_isotropic.md

uv run python scripts/benchmark_jeans_codes.py --anisotropy om --n-stars 4000 --n-bins 12 --engines jeanspy,jeanspy-nuts --seed 52346 --n-walkers 18 --n-steps 120 --n-burn 40 --nuts-warmup 200 --nuts-samples 400 --jeanspy-n-u 96 --report-path benchmarks/jeans_comparison/README_sampler_om.md
```

| Anisotropy | Engine | Sampler | Sampling s | Median predict s | Acceptance | Divergences | q50 log10_rs_pc | q50 log10_rhos | q50 log10_r_a_pc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| isotropic | `jeanspy` | emcee | 1.17 | 0.000672 | 0.702 |  | 3.1431 | -2.0304 |  |
| isotropic | `jeanspy-nuts` | NUTS | 8.86 | 0.000508 | 0.920 | 0 | 3.1356 | -2.0224 |  |
| OM | `jeanspy` | emcee | 1.18 | 0.000558 | 0.617 |  | 3.2355 | -2.1511 | 2.6905 |
| OM | `jeanspy-nuts` | NUTS | 14.6 | 0.000484 | 0.940 | 0 | 3.1632 | -2.0544 | 2.7442 |

The NUTS runs use more warmup than the emcee smoke chains and should not be
read as an equal-proposal comparison. They confirm that the JAX likelihood can
be sampled with gradient-based NumPyro NUTS and gives consistent posterior
regions with no divergences in these runs.

### AGAMA Check

AGAMA was run separately with `--include-slow` because each wrapper evaluation
is much heavier.

| Anisotropy | Engine | Sampling s | Median predict s | Acceptance | q50 log10_rs_pc | q50 log10_rhos | q50 log10_r_a_pc |
|---|---|---:|---:|---:|---:|---:|---:|
| isotropic | `jeanspy` | 0.283 | 0.000658 | 0.590 | 3.0869 | -1.9503 |  |
| isotropic | `jampy` | 1.331 | 0.00270 | 0.588 | 3.1091 | -1.9842 |  |
| isotropic | `agama` | 62.424 | 0.121 | 0.596 | 3.0975 | -1.9646 |  |
| OM | `jeanspy` | 0.320 | 0.000605 | 0.611 | 3.0652 | -1.9269 | 2.8043 |
| OM | `jampy` | 1.626 | 0.00280 | 0.598 | 3.1183 | -1.9985 | 2.8022 |
| OM | `agama` | 72.627 | 0.117 | 0.611 | 3.1109 | -1.9807 | 2.8000 |

## Star Count Check

The default likelihood is a binned velocity-dispersion profile with fixed
`n_bins=12`, so runtime is expected to depend weakly on the raw star count once
the profile has been constructed.

| Stars | Engine | Sampling s | Median predict s |
|---:|---|---:|---:|
| 1000 | `jeanspy` | 0.273 | 0.000566 |
| 1000 | `jampy` | 1.349 | 0.00271 |
| 4000 | `jeanspy` | 0.283 | 0.000658 |
| 4000 | `jampy` | 1.331 | 0.00270 |
| 8000 | `jeanspy` | 0.312 | 0.000616 |
| 8000 | `jampy` | 1.340 | 0.00272 |

For an individual-star likelihood, the same adapters can be reused, but the
likelihood loop should be changed to evaluate all stellar radii rather than the
profile bins.

## Files

- Benchmark CLI: `scripts/benchmark_jeans_codes.py`
- Visualization notebook: `notebooks/benchmark_jeans_codes.ipynb`
- Per-run generated reports: `README_isotropic.md`, `README_om.md`,
  `README_gravsphere_v1_isotropic.md`, `README_gravsphere_v1_om.md`,
  `README_sampler_isotropic.md`, `README_sampler_om.md`,
  `README_scaling_1000.md`, `README_scaling_8000.md`
- Raw artifacts: `artifacts/` (ignored by git)

## Environment Notes

The Python benchmark extras are recorded in `pyproject.toml`:

```bash
uv sync --extra numpyro_cpu --extra benchmark
```

AGAMA is kept out of the project extras because its PyPI build needed local
GSL/Eigen handling in this environment. The working install used the submodule:

```bash
cd external/Agama
uv run python setup.py install --yes
```

The command installs an importable `agama` package into `.venv`. The generated
AGAMA build products remain inside the submodule and are ignored there.
