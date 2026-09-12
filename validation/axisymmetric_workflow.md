# Axisymmetric workflow validation

Recorded 2026-09-12 for the axisymmetric inference extension of PR #63, built
on `17bc27081fecf3ed68a53a3570291d7b351cbd1b`. The model scope and numerical
contracts are in [the API documentation](../docs/axisymmetric.md).

## Initial checks and observed results

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

## Review follow-up: parameter configuration checks

The NumPyro likelihood now checks the combined fixed/sampled physical names
at construction: unknown or missing names, conflicting flattening parameters,
and a missing named velocity mean raise `ValueError` immediately. These checks
share the forward solver's schema and do not evaluate priors or callbacks.
A callable mean does not require `vmem_kms`. When a postprocessor changes the
dictionary, its output is checked before the forward solver runs. Invalid
physical proposals continue to receive log probability minus infinity.

Sixteen regression cases cover these errors, callback preservation, transformed
parameters, JIT/gradients, and postprocessors that add, remove or rename keys.
The full repository suite was rerun after this change with the CPU/float64
environment above: **456 tests and 37 subtests passed in 230.88 s**, including
all **114 axisymmetric tests** and real sampler persistence/restart checks.
There were no skips. The 21 warnings comprised 16 short-chain autocorrelation
warnings, four headless plotting warnings and the existing ensemble walker-count
recommendation. The separate artifact-environment results above predate this
follow-up; the PR CI rebuilds and validates the updated wheel and sdist.

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
changes are below 7.75e-4, within the declared 3e-3 comparison gate.

### Correction of the JAM calculation-path diagnosis

Rechecking the original comparison identified a configuration error in the
benchmark: `analytic_los=True, interp=False` selects **numerical** LOS
integration in jampy 8.1.4. Its constructor overrides `analytic_los` when
`interp=False`. The previous attribution of the 1.27% / 2.46% differences to
an unresolved one-dimensional analytic-kernel integral was incorrect.

The corrected benchmark retains the same MGE coefficients from the fit with
a requested Gaussian budget of 48, as well as the original coordinates,
physics and gravitational-constant conversion. It verifies the
effective path through the returned `vel2` tensor: `None` for analytic LOS,
populated for numerical LOS. The analytic call uses `interp=True`; for ten
positions and no PSF/pixel convolution, its 20-by-10 output-grid threshold
selects direct evaluation of all requested positions. No output interpolation
is introduced by this correction.

Relative differences below are maxima over ten positions in each case,
against the independently integrated analytic kernel, in **second moments**:

| Calculation | Oblate halo Q=0.55 | Prolate halo Q=1.3 |
| --- | ---: | ---: |
| Original call: numerical LOS, 20-by-10 intrinsic grid, 1500 LOS points, epsrel=1e-2 | 1.27e-2 | 2.46e-2 |
| Numerical LOS, 40-by-20 intrinsic grid; other settings unchanged | 2.33e-3 | 4.41e-3 |
| Numerical LOS, 80-by-40 intrinsic grid, 3000 LOS points, epsrel=1e-6 | 5.96e-4 | 1.15e-3 |
| Correct public analytic-LOS path | 7.39e-9 | 1.75e-10 |
| JeansPy, 128 nodes per integral | 1.13e-5 | 2.87e-5 |

The first refinement isolates a contribution from the intrinsic grid. The
last numerical refinement changes several settings together and does not
separate their individual errors. The unchanged JAM kernel is also integrated
using its standard `quad1d`, adaptive quadrature in log(u) with interval and
tolerance refinement, and direct u quadrature with explicit breakpoints. All
agree within the declared 1e-7 kernel-comparison gate. The numerical LOS
results remain recorded as resolution diagnostics; they are not labeled as
analytic-LOS results or as evidence of a failure of JAM's kernel quadrature.

The original report's SHA256 is recorded in `reused_mge_report_sha256`, and
the corrected JSON stores coefficients for both requested Gaussian budgets
(32 and 48), the actual retained component counts, explicit
requested settings, observed path, all comparison outputs and script hash.
The original report remains available in commit `dcaf7e6`. Reproduce the
integration checks without refitting either MGE:

```bash
python scripts/validate_axisymmetric_jam.py \
    --reuse-mge validation/axisymmetric_jam_reference.json \
    --output /tmp/axisymmetric_jam_recheck.json
```

The corrected public analytic path, independently integrated kernel and
finest numerical LOS comparison must meet their recorded gates. The runner
also checks that the numerical differences decrease across these three
resolutions. These tests concern the specified two cases and ten positions;
the MGE-order change is a convergence diagnostic, not a rigorous total-error
bound. These JAM cases use untruncated halos.

The focused recheck (`tests/test_axisymmetric.py` and
`tests/test_source_syntax.py`) passed all 28 tests. The regenerated reference
also passes the public analytic-path agreement check in the ordinary
axisymmetric regression test. Repeating the documented command with both
stored MGE fits reproduced every recorded case result exactly. The original
fine-fit coefficients, log-integrated moments and numerical-LOS output arrays
were also unchanged from the original report. No JeansPy production solver code was changed
for this benchmark correction.

The automated tests also compare to analytic spherical Plummer moments,
existing spherical Jeans and finite-cone J calculations, Poisson/Jeans
residuals, and independent observer-ray J/D quadrature. Matched NumPy/JAX
values alone would not establish physical correctness. The
[`axisymmetric_runtime.json`](axisymmetric_runtime.json) timings likewise do
not establish numerical accuracy or scientific calibration. Parameter-specific
quadrature refinement and inference diagnostics remain necessary for science
analyses; reproduction of the published Hayashi galaxy fits is not claimed.
