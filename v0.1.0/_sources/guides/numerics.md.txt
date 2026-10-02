# Numerical accuracy

The solvers use fixed quadrature rules without an automatic error estimate.
Compare increasing node counts and an independent reference when using a new
parameter domain. See the API reference for each solver's numerical options.

## Axisymmetric quadrature

Force, vertical-pressure and LOS integrals use fixed Gauss–Legendre rules.
Rational maps cover infinite vertical and LOS ranges. LOS quadrature is
centered at maximum tracer density. Defaults
are 96 nodes per integral; numerical settings must be refined for the actual
parameter regime, especially sharp truncation, extreme scale ratios/cusps,
flattening and outer positions. There is no global adaptive accuracy guarantee.
Cost grows approximately as `N_stars*n_force*n_vertical*n_los`. NumPy processes
one sky position at a time; JAX uses `lax.map` and rematerialization to bound
intermediate memory during parameter differentiation.

The three node counts control the [force and Jeans integrals](../theory.md#axisymmetric-jeans-equations).
J/D factors have [separate quadrature settings](factors.md#axisymmetric-finite-cone-factors).

## Precision and physical-parameter gradients

Use the [backend tutorial](../tutorials/backends.ipynb) to configure JAX,
inspect the effective device/precision and measure synchronized execution.
CPU float64, CPU float32 and actual GPU conditions need separate accuracy
assessments. Check derivatives against finite differences at several step
sizes and quadrature orders over the intended parameter domain. The
[model contract](contracts.md) describes differentiability boundaries.

## Specialized spherical kernels

The specialized hypergeometric and fixed-$\eta=2$ Baes kernels support the
spherical JAX solver. They are real-valued approximations with restricted
domains; they do not replace a general complex special-function library.
The following check uses SciPy and mpmath references in a small regular domain.

```{literalinclude} ../../../examples/docs_numerics.py
:language: python
:start-after: special-functions-start
:end-before: special-functions-end
```

`baes_eta2_kernel_jax` clamps some intermediate arguments and replaces
nonfinite values. Validate physical parameters separately: a finite low-level
kernel output does not certify a valid anisotropy model.
