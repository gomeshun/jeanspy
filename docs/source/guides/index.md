# Topic guides

Use these references when choosing a model or checking a calculation's
conventions and limits. Each guide covers a specific topic:

| Guide | What to look up |
| --- | --- |
| [Model contracts](contracts.md) | Units, array shapes, parameter mutation and error behavior across backends |
| [Profile conventions](profiles.md) | Tracer scale definitions, Sersic deprojection domains and halo cutoffs |
| [Axisymmetric geometry](axisymmetric.md) | Coordinates, flattening, inclination and geometry-specific inference constraints |
| [J and D factors](factors.md) | Aperture definitions, finite-cone integrals and cusp restrictions |
| [Numerical accuracy](numerics.md) | Quadrature refinement, gradient checks and specialized kernel limits |
| [API migration](api-migration.md) | Explicit backend names, density cutoffs, precision and sampled-parameter mappings |

For setup and executable workflows, follow the [tutorials](../tutorials/index.md),
including [JAX and backends](../tutorials/backends.ipynb),
[MCMC and diagnostics](../tutorials/inference.ipynb), and
[saving and resuming](../tutorials/storage.ipynb).
[Theory](../theory.md) collects the Jeans and likelihood equations;
the [API reference](../api/index.md) documents individual calls.

```{toctree}
:maxdepth: 1
:hidden:

contracts
profiles
axisymmetric
factors
numerics
api-migration
```
