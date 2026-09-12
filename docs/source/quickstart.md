# Quickstart

The following examples execute in the documentation CI. Distances are in pc,
velocities in km/s, masses in solar masses, and halo densities in
solar masses per cubic parsec. The spherical LOS API requires `R_pc > 0`.

## Spherical prediction

```{literalinclude} ../../examples/docs_spherical.py
:language: python
:start-after: example-start
:end-before: example-end
```

The model combines a normalized Plummer tracer, an NFW dark halo and constant
spherical anisotropy. The returned variance has the same shape as `R_pc`.
The stellar tracer supplies weights and contributes no gravitational mass.

## Axisymmetric prediction

```{literalinclude} ../../examples/docs_axisymmetric.py
:language: python
:start-after: example-start
:end-before: example-end
```

The model uses cylindrical alignment and constant `beta_z`. With zero mean
streaming, the LOS second moment is a variance. Sky coordinates are signed;
`x_pc` lies along the line of nodes. Inclination is in radians.

## A physical-parameter gradient

```{literalinclude} ../../examples/docs_jax.py
:language: python
:start-after: example-start
:end-before: example-end
```

Differentiate the JAX forward model or likelihood in the interior of the
admissible parameter domain. Quadrature orders and model-family choices are
static settings. The [J/D factor functions](guides/factors.md) are NumPy
postprocessing and do not participate in this JAX gradient.
