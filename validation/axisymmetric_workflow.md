# Axisymmetric workflow validation

Recorded 2026-09-12 for the axisymmetric inference extension of PR #63, built
on `17bc27081fecf3ed68a53a3570291d7b351cbd1b`. The model scope and numerical
contracts are in [the API documentation](../docs/axisymmetric.md).

## Checks and observed results

| Check | Result |
| --- | --- |
| Full repository suite, including opt-in MCMC | 440 passed; 37 subtests passed; 229.82 s |
| Axisymmetric tests | 98 passed; 53.68 s |
| Classical executable example, run twice | 10 then 14 stored steps, 12 walkers; second run resumed; 96 posterior rows after discarding six warmup steps |
| NumPyro executable example, run twice | Four then eight posterior draws; second run resumed; no divergences in these short runs |
| Wheel and sdist, installed in separate base environments | README quick start, packaged data, public axisymmetric inference and J/D smoke checks passed; JAX/NumPyro/Matplotlib absent |
| Wheel and sdist, installed in separate NumPyro CPU environments | Spherical and axisymmetric likelihood/gradient smoke checks passed |
| Dependency compatibility | `uv pip check` passed in all four artifact environments |

The full suite used Python 3.12.13, NumPy 2.4.3, SciPy 1.17.1, JAX/JAXlib
0.9.1 and NumPyro 0.20.0 on CPU with float64. It emitted 20 warnings: 15
short-emcee-chain autocorrelation warnings, four headless notebook plotting
warnings, and one existing ensemble-sampler walker-count warning. No tests
were skipped in this run. Persistence tests exercise real emcee and NUTS
chains, h5netcdf and Zarr, and reject changed analysis identity before
appending to stored data.

Artifact checks ran outside the checkout against independently installed
wheel/sdist packages, with imports verified to come from their respective
`site-packages`. Fresh base environments resolved NumPy 2.5.3, SciPy 1.18.1,
pandas 3.0.5 and h5py 3.16.0. CPU environments additionally resolved JAX/JAXlib
0.11.1, NumPyro 0.21.0, ArviZ 1.3.0 and Zarr 3.3.0. These checks do not cover
GPU execution. The CI matrix covers Python 3.12/3.13, locked/lowest direct
dependencies and Windows/macOS/Linux base environments, with fresh resolutions
and opt-in MCMC in the release-validation matrix.

## Reproduction

From an installed development checkout with the optional CPU dependencies:

```bash
JAX_PLATFORMS=cpu JEANSPY_JAX_ENABLE_X64=true python -m pytest -q --run-mcmc

for backend in classical numpyro; do
  for run in 1 2; do
    JAX_PLATFORMS=cpu JEANSPY_JAX_ENABLE_X64=true \
      python examples/axisymmetric_inference.py --backend "$backend" \
      --output-dir "/tmp/axisymmetric-validation-$backend" \
      --nodes 16 --warmup 6 --draws 4
  done
done

uv build --no-sources
```

Use new output directories for the initial example runs. Each run writes the
observations, explicit sampling-coordinate priors, posterior table, a small
deterministic subset of posterior J/D factors, and a summary with restart and
sampler diagnostics. The example uses five free parameters and eight synthetic
stars; the 16-node, short-chain settings above exercise the persistence and
export workflow. They do not support a convergence, coverage, calibration or
parameter-recovery claim.

The artifact smoke entry point is
[`scripts/validate_release_artifact.py`](../scripts/validate_release_artifact.py):
run `--base --readme /path/to/README.md` and `--numpyro-cpu` with a fresh
environment's interpreter from outside the checkout. Set `GITHUB_WORKSPACE`
to the checkout path to enforce the installed-artifact import check. The
existing release workflow performs these steps before any publication.

## Independent numerical evidence and limits

[`axisymmetric_jam_reference.json`](axisymmetric_jam_reference.json) preserves
two oblate/prolate comparisons against the independent Cappellari (2008) JAM
analytic LOS kernel, MGE coefficients, kernel/source hashes and refinement
results. Relative second-moment differences are 1.13e-5 and 2.87e-5; MGE-order
changes are below 7.75e-4, within the declared 3e-3 comparison gate. The raw
high-level JAM outputs differ by up to 1.27% and 2.46% because its default
quadrature does not resolve these broad MGEs. Those outputs are retained as a
failed comparison, and the unchanged kernel is resolved separately using
adaptive quadrature in log(u), with interval/tolerance refinement.

The automated tests also compare to analytic spherical Plummer moments,
existing spherical Jeans and finite-cone J calculations, Poisson/Jeans
residuals, and independent observer-ray J/D quadrature. Matched NumPy/JAX
values alone would not establish physical correctness. The
[`axisymmetric_runtime.json`](axisymmetric_runtime.json) timings likewise do
not establish numerical accuracy or scientific calibration. Parameter-specific
quadrature refinement and inference diagnostics remain necessary for science
analyses; reproduction of the published Hayashi galaxy fits is not claimed.
