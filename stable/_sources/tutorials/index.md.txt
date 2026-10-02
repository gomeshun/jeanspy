# Tutorials

Start with the [Quickstart](../quickstart.ipynb) for one complete JAX/NumPyro
analysis. These chapters explain each part of that workflow, in order.
Use the [API reference](../api/index.md) to look up a specific name or signature.

| Step | What you will learn |
| --- | --- |
| [1. Units and conventions](units.ipynb) | Prepare radii, velocities and errors; distinguish tracer number from mass |
| [2. Choose a backend](backends.ipynb) | Switch NumPy/JAX interfaces, measure execution time and use automatic differentiation |
| [3. Define a model](models.ipynb) | Compose components, set parameters and implement a custom tracer |
| [4. Predict observables](predictions.ipynb) | Evaluate densities, mass and LOS dispersion; plot and generate mock data |
| [5. Run MCMC](inference.ipynb) | Define priors, sample, check R-hat/ESS and NUTS divergences, and read corner contours |
| [6. Save and resume](storage.ipynb) | Export draws, preserve chain structure and restart the same analysis |

```{toctree}
:hidden:
:maxdepth: 1

units
backends
models
predictions
inference
storage
```

## Worked analyses

Apply the steps to complete spherical and axisymmetric analyses. The spherical
example includes recorded outputs and their execution metadata. Both are
short workflow examples; their completion does not demonstrate convergence
or calibration.

```{toctree}
:maxdepth: 1

spherical
axisymmetric
```
