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
published versions and significant changes.

Start with the [installation guide](installation.md), then follow the path
that matches what you need:

| Section | Purpose |
| --- | --- |
| [Quickstart](quickstart.ipynb) | Define a model, plot a prediction and mock data, run NumPyro NUTS and inspect the result |
| [Tutorials](tutorials/index.md) | Learn units, backends, model construction, predictions, MCMC and storage step by step |
| [API reference](api/index.md) | Look up every public class, function, method and property by name |
| [Topic guides](guides/index.md) | Consult geometry, profile, numerical and J/D-factor details |

The [model contract](guides/contracts.md) defines units, geometry, array shapes
and the limits of differentiation.

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
tutorials/index
api/index
guides/index
theory
citing
references
```
