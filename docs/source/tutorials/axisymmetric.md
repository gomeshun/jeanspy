# Axisymmetric synthetic analysis

This example samples scale radius, density, cylindrical anisotropy,
inclination and systemic velocity, and computes finite-cone J/D factors from
a deterministic subset of saved posterior draws. Priors are explicit in
`log10_rhos_Msunpc3`, `log10_rs_pc`, `bfunc_beta_z`, `cos_inclination` and
`vmem_kms`.

Apply the [six tutorial steps](index.md) to a flattened system. Unlike the
spherical Quickstart, this complete script also handles projected flattening,
inclination and posterior J/D-factor calculation.

## 1. Define the geometry and observations

The tracer is an oblate Plummer component, and the halo is a spheroidal Zhao
profile. The example fixes the projected tracer axis ratio and converts it
to an intrinsic ratio for each inclination. Positions are signed `x_pc`,
`y_pc` coordinates in pc. The velocity likelihood conditions on those
positions; it does not infer the spatial selection function.
The [geometry reference](../guides/axisymmetric.md#units-geometry-and-physical-parameters)
defines inclination, deprojection and the physical parameter domains.

## 2. Choose priors and run a backend

The prior table explicitly distinguishes logarithmic scales, transformed
cylindrical anisotropy and cosine inclination. A uniform prior in cosine
inclination is not uniform in inclination. The two commands below use the
same physical configuration and mock-data recipe.

```bash
python examples/axisymmetric_inference.py \
    --backend classical --output-dir /tmp/axisymmetric-emcee
JEANSPY_JAX_ENABLE_X64=true python examples/axisymmetric_inference.py \
    --backend numpyro --output-dir /tmp/axisymmetric-nuts
```

## 3. Save, resume and inspect

Repeat the same command to resume. Keep the same warmup setting when resuming
the classical example so that its export discards the original warmup steps.
Outputs include `observations.csv`, `prior.csv`, `posterior.csv`,
`derived_factors.csv`, `summary.json` and persisted sampler state.

Default settings use eight stars and short chains. They exercise the complete
storage path; the summary explicitly records that convergence and calibration
are unestablished.

Inspect the saved chain as described in the [MCMC tutorial](inference.ipynb)
before interpreting derived factors. The J/D factors are NumPy postprocessing
of a deterministic subset of posterior draws, not differentiable outputs
of the NUTS model. Their cone aperture and distance are part of the analysis.
See the [J/D-factor guide](../guides/factors.md#axisymmetric-finite-cone-factors)
for cutoff, cusp and quadrature requirements.

````{dropdown} Complete executable script
```{literalinclude} ../../../examples/axisymmetric_inference.py
:language: python
```
````

## Forward predictions and gradients

For a smaller calculation before running inference, this example evaluates
an axisymmetric JAX prediction and its derivative with respect to the halo
density scale. It checks the derivative against the known linear scaling.
Follow the [backend tutorial](backends.ipynb) for runtime configuration and
the [numerical accuracy guide](../guides/numerics.md) for refinement checks.

```{literalinclude} ../../../examples/docs_jax.py
:language: python
```
