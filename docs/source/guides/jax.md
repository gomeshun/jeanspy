# JAX, precision and gradients

Configure `JEANSPY_JAX_PLATFORM=cpu` and `JEANSPY_JAX_ENABLE_X64=true` before
starting a reference calculation. Inspect
`jeanspy.model_numpyro.get_runtime_config()` and `jax.devices()` rather than
inferring the device from installed packages.

The functional spherical models and the axisymmetric JAX solver can be JIT
compiled and differentiated with respect to physical parameters. For
spherical anisotropy, choose the JAX kernel path when gradients are required;
a SciPy callback path is not an all-JAX differentiable calculation. The
axisymmetric solver uses JAX operations and bounded-memory mapping over sky
positions. Node counts, backend selections and array shapes affect compilation.

The spherical components receive physical parameters explicitly on each call:

```{literalinclude} ../../../examples/docs_jax_spherical.py
:language: python
:end-before: spherical-jax-end
```

Functional density, mass and anisotropy calculations use the same convention:

```{literalinclude} ../../../examples/docs_jax_spherical.py
:language: python
:start-after: functional-profiles-start
:end-before: functional-profiles-end
```

For an axisymmetric physical-parameter derivative:

```{literalinclude} ../../../examples/docs_jax.py
:language: python
```

This example checks the density derivative against linear scaling. The
benchmark gradient suite additionally compares finite differences at several
step sizes and quadrature orders. Agreement at a single point is insufficient
to establish a whole prior domain.

JAX dispatch is asynchronous. End timed predictions and gradients with
`jax.block_until_ready(result)`. Record host preprocessing, transfers, first
compilation/execution and subsequent execution separately. CPU float64,
CPU float32 and actual GPU conditions are separate experiments. Lower
precision requires its own numerical-accuracy assessment.

Posterior density and mass calculations can use the JAX model where supported.
Current J/D factors use NumPy quadrature after sampling; their availability
does not imply differentiability through those factors.
