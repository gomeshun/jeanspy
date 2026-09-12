# Reproducible dynamical inference with JeansPy

JeansPy solves spherical and axisymmetric Jeans equations for dwarf spheroidal
galaxies. Build a tracer, halo and anisotropy model; predict velocity second
moments; fit explicit priors and likelihoods with emcee or NumPyro; and retain
the inputs and diagnostics needed to reproduce the analysis.

This is the **development documentation**, based on the public API in commit
[`1a0ad40`](https://github.com/gomeshun/jeanspy/tree/1a0ad4028d26af1df389ebdfdf992285ec50f8bb).
No stable documentation version has been designated. Package version `0.1.0`
is alpha software; the development branch includes additions that may not be
present in an installed PyPI distribution.

Start with the [installation guide](installation.md) and
[executable quickstart](quickstart.md). The [model contract](guides/contracts.md)
defines units, geometry, array shapes and the limits of differentiation.

| Task | Start here |
| --- | --- |
| Predict spherical velocity dispersions | [Quickstart](quickstart.md) |
| Model flattened systems | [Axisymmetric guide](guides/axisymmetric.md) |
| Use physical-parameter gradients | [JAX and precision](guides/jax.md) |
| Specify priors, fit and resume | [Inference and storage](guides/inference.md) |
| Calculate annihilation/decay factors | [J and D factors](guides/factors.md) |
| Assess the evidence for a result | [Validation](validation/index.md) |

```{toctree}
:maxdepth: 2
:caption: Learn and use

installation
quickstart
theory
guides/index
tutorials/index
api/index
```

```{toctree}
:maxdepth: 2
:caption: Evidence and project

validation/index
comparison/index
citing
references
changelog
development
```
