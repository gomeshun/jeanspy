# Zhao mass derivative stability

The change recovers weak scale-radius derivatives in small-radius cores by
rearranging the enclosed-mass prefactor **before automatic differentiation**.
It does not solve general Fisher-matrix degeneracy. This directory retains both
the successful analytic regressions and the remaining nearly singular failures.

## What changes

Write `p = 3 - gamma`, `x = min(r, r_t)/r_s`. For `x < 1`, the original
quadrature prefactor contains `r_s^3 * x^p`. Differentiating its factors
separately generates order-one logarithmic derivatives `3` and `-p`.
At `gamma = 0`, they should cancel exactly, leaving only the much smaller
derivative of the density transition. Floating-point roundoff can overwhelm
that remaining derivative and even reverse its sign.

The revised inner branch combines the prefactor as

```text
exp(p * log(min(r, r_t)) + gamma * log(r_s)).
```

Its explicit scale dependence is `gamma`, with no subtraction of `3` and `p`.
A single exponential also avoids the spurious `0 * inf` that would occur in
`r^3 * x^(-gamma)` for sufficiently small radii and steep cusps. The quadrature,
outer branch, physical model, parameter coordinates, and Jeffreys-prior
mathematical definition are unchanged. No eigenvalue floor, rank cutoff, jitter,
or custom derivative is introduced.

## Independent analytic regression

For `alpha = p`, `beta = 3 + p`, at fixed other parameters,

```text
M(r) = 4*pi*rho_s*r_s^3/p * x^p/(1 + x^p)
d log M / d log r_s = gamma + p*x^p/(1 + x^p).
```

For a core (`gamma=0, alpha=3, beta=6`), the second scale derivative is
`-9*x^3/(1+x^3)^2`. These exact expressions supply references independently of
both the old code and finite differencing.

With `r_s=1000 pc`, `rho_s=0.01 Msun/pc^3`, no truncation, JAX 0.9.1 and float64,
the relative errors in the first scale derivative are:

| Radius / pc | Exact derivative (approximately) | CPU before | CPU after | GPU before | GPU after |
|---:|---:|---:|---:|---:|---:|
| 0.000001 | 3e-27 | 4.68e10 | 1.42e-14 | 1.45e11 | 1.47e-14 |
| 0.001 | 3e-18 | 98.8 | 1.29e-14 | 290 | 1.33e-14 |
| 0.1 | 3e-12 | 1.15e-4 | 1.49e-14 | 6.81e-5 | 1.53e-14 |
| 1 | 3e-9 | 4.38e-8 | 1.33e-14 | 3.00e-9 | 1.38e-14 |
| 10 | 3e-6 | 4.39e-11 | 1.42e-14 | 1.27e-10 | 1.40e-14 |

At `r=0.001 pc`, the old CPU/GPU results are negative although the exact
result is positive. These are relative errors in a very small derivative;
they do not imply comparably large errors in the mass or observable dispersion.

The 12 new tests cover forward/reverse AD under JIT, float32/float64,
`gamma in {0, 1e-20, 0.5, 1}`, the scale-radius transition, truncation,
the second derivative, the shared axisymmetric implementation, and steep-cusp
prefactor overflow/underflow. Relative tolerances for the scale derivative are
`2e-12` in float64 and `2e-5` in float32, with **zero absolute tolerance**.
Float32 is tested down to `r/r_s=1e-6`; at still smaller ratios the raw derivative
of mass can underflow before `log(M)` rescales it. This fix does not extend the
floating-point exponent range.

## Ordinary-case consistency

The audit compares the pre-change implementation at commit
`d4bace441fd4d1d5b39c11de68ff89e410542ffa` against this source. The 48 parameter
sets use `alpha in {0.5,1,3,5}`, `beta in {2,3,6,10}`, and
`gamma in {0,1,2.9}`. Seven radii span `r/r_s = 0.01,0.1,0.9,1,1.1,10,100`,
with `r_s=1000 pc`, `rho_s=0.01`, and `r_t=50000 pc`.

- Maximum relative mass change: `3.56e-15` (JAX CPU and GPU), `2.23e-15` (NumPy).
- Maximum absolute change in all six dimensionless mass scores: `7.11e-15`.
  Physical derivatives of `log(M)` are scaled by `r_s`, `rho_s`, and `r_t`;
  the three dimensionless shape derivatives are unscaled.
- At the core/cusp generating truths of the two 64-star mocks, the joint
  Jeffreys log-prior changes by less than `3e-12`; total variances and scores
  change by less than `1e-15`.

These are finite-grid consistency checks, not universal error bounds.

## Remaining nearly singular Fisher cases

`fisher_cases.json` contains synthetic radii, velocity uncertainties, and the
six structural coordinates used in the original Segue-like precision audit.
For each mock, `re=29 pc`, the mass and kernel Gauss orders are 128, the LOS
quadrature has 128 nodes, and `u_max=1e5`. The score matrix is
`J_ij = d log(S_i)/d theta_j / sqrt(2)`, where `S_i` is the total LOS variance.
A column-scaled QR gives its log-volume. The joint prior includes the additional
systemic-velocity factor `0.5*log(sum(1/S_i))`.

Absolute log-prior errors against an independent binary128 implementation of
that **fixed quadrature target** are:

| Case | CPU before | CPU after | GPU before | GPU after |
|---|---:|---:|---:|---:|
| Prior edge 22 | 1.5403e-3 | 5.1457e-5 | 9.2367e-4 | 7.3292e-5 |
| Prior edge 31 | 1.9636e-6 | 1.9636e-6 | 3.7062e-5 | 3.7062e-5 |

The first point improves in this build, but **neither point meets the original
absolute log-prior tolerance of 1e-6 on both devices**. The second point samples
only the unchanged outer branch. Near-linear dependence between score columns
still amplifies roundoff, and small arithmetic/compiler changes can move these
errors in either direction. This change does not replace high-precision
fallbacks or establish posterior/calibration accuracy. Quadrature convergence
is a separate question from the arithmetic comparisons reported here.

## Reproduce

Use a checkout that contains the baseline commit (fetch it if the clone is
shallow), plus the `numpyro_cpu` and `dev` extras. From the repository root:

```sh
JAX_PLATFORMS=cpu uv run --extra numpyro_cpu --extra dev python -m pytest -q tests/test_zhao_small_radius_gradients.py
JAX_PLATFORMS=cpu uv run --extra numpyro_cpu --extra dev python scripts/validate_zhao_small_radius_derivatives.py --output /tmp/zhao-cpu.json
```

In an environment with the CUDA JAX extra installed, repeat with
`JAX_PLATFORMS=cuda` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
The archived `cpu.json` and `gpu.json` include the source hashes, dependency
versions, and complete before/after values. GPU: NVIDIA RTX 3090.

For the slower independent reference, GCC 13.3/libquadmath was used. The C++
source propagates six dual-number derivatives in IEEE binary128 through the
frozen mass, anisotropy kernel, and LOS quadrature. It evaluates log-volumes by
both Householder QR and twice-reorthogonalized Gram-Schmidt. Inputs and Gauss
nodes are binary64, promoted before computation; recorded log-priors are rounded
back to binary64.

```sh
g++ -O2 -fPIC -shared validation/zhao_derivative_stability/reference.cpp -lquadmath -o /tmp/zhao-reference.so
uv run python validation/zhao_derivative_stability/check_reference.py --library /tmp/zhao-reference.so --output /tmp/zhao-reference-check.json
```

`reference_check.json` records reproduction of the four saved references and
agreement between the two QR calculations. This reference is an optional
validation artifact, not a runtime dependency or an inference fallback.

## Test results for this revision

- CPU full default suite: **533 passed, 34 skipped, 37 subtests passed**
  (`python -m pytest -q`, 222.82 seconds). The default suite does not enable the
  opt-in MCMC execution tests; four warnings concern noninteractive notebook
  figures.
- GPU: **12 passed** for `tests/test_zhao_small_radius_gradients.py`
  (15.40 seconds).
- The independent binary128 run reproduced all four saved values exactly after
  rounding to binary64. Its two QR routes differed by at most `4.80e-22` in
  log-volume (44.27 seconds for all four points).
