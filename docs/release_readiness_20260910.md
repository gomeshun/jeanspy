# Release readiness: issues #53–#57

> Historical record for PR #61. The subsequent audit and its fixes are tracked
> in [the release audit remediation record](release_audit_fixes_20260910.md).

This work addresses the five open correctness/release issues other than
[axisymmetric modeling #52](https://github.com/gomeshun/jeanspy/issues/52).
It extends [PR #61](https://github.com/gomeshun/jeanspy/pull/61), starting from
`d7b62aa5e91bfeaa6b422004436ea6e5888690bb` over main
`3f46feba7c100a5a21856185044c097a8f67b632`. The numerical/inference implementation
and regression tests described here are recorded in `ecba41e` and its parents.
CI wiring and documentation follow that commit.

## Decisions agreed with the maintainer

- Existing LOS solvers support only finite `R>0`. No arbitrary epsilon or
  unvalidated central-limit approximation is introduced. Classical calls
  reject invalid inputs; JAX returns per-element NaNs consistently under JIT.
- Default classical inference uses log10 scale/density coordinates and
  `bfunc_beta_ani = log10(1-beta_ani)`, with explicit finite bounds. A missing
  config creates an unfilled template and fails early. Supplied systemic
  velocity priors survive data resets; empirical bounds require opt-in.
- Release validation requires full tests, including MCMC, with both locked
  and freshly resolved dependencies on Python 3.12 and 3.13. Ordinary CI
  remains locked and skips opt-in MCMC tests.

## Requirement audit

| Issue | Implemented behavior | Evidence |
| --- | --- | --- |
| [#53](https://github.com/gomeshun/jeanspy/issues/53) | Regularized Zhao mass supports finite-radius/truncated `b<=3` profiles with `a>0`, `g<3`; invalid mass cannot be sanitized into zero LOS variance. Explicit JAX analytic evaluation falls back outside the beta-function domain. | `tests/test_zhao_mass_domain.py`: independent quadrature, NFW/non-NFW `b=3`, core/cusp and truncation cases, scalar/vector inputs, invalid mass and likelihood checks. Issue reproduction gives `33785.62865245551 Msun` at 100 pc. |
| [#54](https://github.com/gomeshun/jeanspy/issues/54) | Power-coordinate integration removes the central cusp without a cutoff; the outer segment uses log radius. Public `n_steps` / `dm_mass_n_steps` controls separate mass resolution from Jeans resolution. | Same test module: Dehnen closed forms through `g=2.99`, independent mass grid, float32/64, JIT and finite-difference gradient checks, independent mass/LOS refinement. The `g=2.5` reproduction matches `177715317.52633464 Msun`, replacing a roughly 132% overestimate. |
| [#55](https://github.com/gomeshun/jeanspy/issues/55) | Prior coordinate names/order/count and finite bounds are validated before inference. Initial photometry draws respect the finite support. Data reset preserves explicit priors or synchronizes empirical bounds, and invalidates WBIC temperature. | `tests/test_classical_inference_workflow.py`, `tests/test_shared_data_reset.py`, and `scripts/example_classical_inference.py`: real Plummer+NFW model, finite posterior, emcee, chain/blobs, storage and restart with reconstructed models. Malformed templates/schema/bounds and failed resets are tested. |
| [#56](https://github.com/gomeshun/jeanspy/issues/56) | Scalar/nonempty 1-D shape contract and finite positive radii are enforced in all LOS entry points. Invalid array members do not affect valid JAX predictions. Nonfinite integrands/results are preserved as errors/NaNs; only known classical endpoint/underflowed-tracer zeros are discarded. Abel weights avoid differentiating `acosh(1)` in inactive branches. | `tests/test_los_radius_domain.py`: zero/negative/NaN/infinite radii, scalar/mixed arrays, float32/64, eager/JIT/direct methods, invalid likelihood rejection, independent near-center integration and radial-gradient finite differences. |
| [#57](https://github.com/gomeshun/jeanspy/issues/57) | Release calls the local reusable test workflow at the exact caller commit. `publish.needs` contains both `build` and `tests`. Both Python versions and both dependency modes run full MCMC-inclusive tests. Artifact isolation and exact artifact reuse are retained. | `.github/workflows/test.yml`, `.github/workflows/release.yml`, `RELEASE.md`; actionlint 1.7.12 passes. Local runtime/artifact checks are listed below; GitHub checks on PR #61 exercise the reusable release matrix without publishing. |

## Local runtime and artifact validation

Every full-suite invocation uses `python -m pytest tests --run-mcmc -q` on
CPU. The existing developer `.venv` also passed, but contained additional
packages and differing auxiliary versions, so it is not used as proof of
the exact locked environment.

| Environment | Result |
| --- | --- |
| Python 3.12.13, exact `uv.lock` | 275 passed, 37 subtests passed |
| Python 3.13.12, exact `uv.lock` | 275 passed, 37 subtests passed |
| Python 3.12.13, fresh dependency resolution | 275 passed, 37 subtests passed |
| Python 3.13.12, fresh dependency resolution | 275 passed, 37 subtests passed |

The locked environments were independently created using
`UV_PROJECT_ENVIRONMENT=<new directory> uv sync --locked --python <version>
--extra numpyro_cpu --extra dev --extra plotting`. Fresh environments were
created with `uv venv` and `uv pip install -e '.[numpyro_cpu,dev,plotting]'`.
The reusable CI uses noneditable installs for its fresh environments.

| Package | Locked | Fresh (2026-09-10) |
| --- | --- | --- |
| NumPy | 2.4.3 | 2.5.3 |
| SciPy | 1.17.1 | 1.18.1 |
| JAX | 0.9.1 | 0.11.1 |
| NumPyro | 0.20.0 | 0.21.0 |
| ArviZ | 1.0.0 | 1.3.0 |
| xarray | 2026.2.0 | 2026.7.0 |
| Zarr | 3.1.5 | 3.3.0 |
| pytest | 9.0.2 | 9.1.1 |

Additional checks completed:

- `uv build --no-sources` and `twine check` pass for wheel and sdist.
- Separate new Python 3.12 environments install each built artifact and
  execute the README and runtime-data checks. `GITHUB_WORKSPACE` is set so
  the validator rejects imports from the source checkout.
- The wheel's `numpyro_cpu` extra passes the installed-artifact Jeans
  likelihood/JAX smoke test with the fresh versions above.
- Both `numpyro_cpu` and `numpyro_cuda12` resolve from both built artifacts
  in `uv pip install --dry-run`; no CUDA execution is claimed.
- The classical example writes a 12-step, 16-walker, 6-parameter chain and
  confirms finite log probabilities and exact preservation of the first
  eight steps across restart. Seed 55, synthetic observations and explicit
  example priors are recorded by the script.
- `actionlint .github/workflows/test.yml .github/workflows/release.yml` passes
  with the checksum-verified official actionlint 1.7.12 binary.

## Numerical validation and limits

The Zhao grid spans `a={0.5,1,3,5}`, `b={2,3,4,8}`, `g={0,1,2.5,2.99}` and
`r/rs={1e-6,0.01,1,100,1e6}`. Independent QUADPACK mass references meet the
test tolerances of `1e-6` (float64) and `5e-5` (float32). Shape-parameter
gradients are supported on auto/numeric paths; the explicit incomplete-beta
path retains its documented JAX shape-autodiff limitation.

For the near-center test, a Plummer tracer (`Re=200 pc`), NFW halo
(`rs=1000 pc`, `rho_s=0.01 Msun/pc^3`, `r_t=10000 pc`) and isotropy are held
fixed. Swapping the isotropic Jeans and projection integrals gives

`sigma_los^2(R) = 2 G / Sigma(R) integral_R^inf nu(r) M(r) sqrt(r^2-R^2)/r^2 dr`.

The reference uses adaptive integration in physical LOS distance
`z=sqrt(r^2-R^2)`, independently of the production kernels and fixed grids.
Its central limit is `21.703164955165118 (km/s)^2`; the API intentionally
rejects `R=0`. At `R={0.001,0.01,0.1,1,50,300} pc`, measured float64 maximum
relative errors against this reference are:

| Solver | Maximum relative error |
| --- | --- |
| Classical DE quadrature, defaults | 1.12e-9 |
| JAX kernel, `u_max=1e8`, `n_u=1025` | 2.08e-10 |
| JAX Abel, `u_max=1e8`, `n_r=8192` | 9.12e-5 |

The tests also increase the outer limit and resolution to check convergence.
These special near-center settings do not broaden the documented default
`R/Re>=0.005` accuracy envelope or establish general Zhao LOS accuracy.
The existing default accuracy-contract and eta2 stress benchmarks also
complete successfully. On the eight-case default contract set, maximum
relative errors are `3.697e-4` (CPU float64), `3.716e-4` (CPU float32) and
`3.762e-4` (GPU-oriented float32 grid on CPU), all below `1e-3`.
GPU-oriented float32 grids are CPU arithmetic proxies only. The eta2 script
also reports deliberately coarse/legacy grids; those larger errors are not
results for the maintained default settings.

These are numerical and software-workflow checks. They do not establish
posterior coverage, scientific prior calibration, or a physically admissible
distribution function for every tracer/halo/anisotropy combination.

## Failures found during validation

- PR #61's original artifact job failed at `numpyro.factor`: importing a
  NumPyro submodule before materializing ArviZ's lazy parent could leave
  `numpyro.distributions.distribution` unavailable. The artifact validator
  now imports distributions via the parent. Reproduction failed before and
  passes after, including the isolated built-wheel test.
- The new positive-radius gradient test exposed NaN Abel derivatives from
  inactive `acosh(1)` branches. Safe inactive arguments fix the derivative
  without changing active bin weights; finite-difference comparison passes.
- An isolated eta2 test exposed an inherited Baes warning that tried to read
  a nonexistent free `eta` parameter. Fixed-eta models now bypass that warning;
  the test passes both alone and in the full suite.
- The local execution sandbox stalled even a standalone `zarr.open_group`.
  Running the same minimal example outside that restriction completed
  immediately. Full storage/MCMC checks therefore run outside that restriction;
  no Zarr implementation change or version pin was introduced to hide it.

The release workflow follows GitHub's documented
[local reusable-workflow commit semantics](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows)
and keeps all PyPI writes confined to the tag-only publish job. No release tag
or package publication is part of this issue-resolution work.
