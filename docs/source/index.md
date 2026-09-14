# Reproducible dynamical inference with JeansPy

JeansPy solves spherical and axisymmetric Jeans equations for dwarf spheroidal
galaxies. Build a tracer, halo and anisotropy model; predict velocity second
moments; fit explicit priors and likelihoods with emcee or NumPyro; and retain
the inputs and diagnostics needed to reproduce the analysis.

```{only} development
This is the **development documentation**. It may describe changes beyond the
latest published package; use the source checkout in the installation guide.
```

```{only} release
This is the documentation for **JeansPy @PACKAGE_VERSION@**. Use the matching
package version in the installation guide when reproducing these examples.
```

Documentation version: `@DOCS_VERSION@`. Source reference:
[`@SOURCE_LABEL@`](https://github.com/gomeshun/jeanspy/tree/@SOURCE_REF@).
The [release history](https://github.com/gomeshun/jeanspy/releases) identifies
published versions; validation results apply to their stated models and domains.

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
