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

Start with the [installation guide](installation.md) and
[executable quickstart](quickstart.md). The [model contract](guides/contracts.md)
defines units, geometry, array shapes and the limits of differentiation.

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
theory
guides/index
tutorials/index
api/index
citing
references
```
