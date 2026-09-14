# Numerical accuracy

The solvers use fixed quadrature rules without an automatic error estimate.
Compare increasing node counts and an independent reference when using a new
parameter domain. See the API reference for each solver's numerical options.

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
