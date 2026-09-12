# Numerical helpers

The classical double-exponential quadrature takes a vectorized function and
fixed nodes. It returns an integral without an error estimate. Compare several
node counts and an independent reference before adopting it for a new domain.

```{literalinclude} ../../../examples/docs_numerics.py
:language: python
:end-before: quadrature-end
```

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
kernel output does not certify a valid anisotropy model. Likewise, the
`xp` argument of classical `dequad` is not a promise of JAX tracing through
node generation or physical parameters.
