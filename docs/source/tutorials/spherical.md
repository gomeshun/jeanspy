# Spherical synthetic analysis

The existing base-package example generates velocities from a Plummer tracer
and NFW halo, records explicit priors, samples with emcee, saves HDF5 draws and
checks that reopening and appending preserves the original chain.

```bash
python scripts/example_classical_inference.py --output-dir /tmp/spherical-example
```

Use a new output directory. Inputs and outputs include `observations.csv`,
`prior.csv`, the persisted sampler state and `result.json`. The seed is 55.
The displayed run contains only 12 stored steps and does not establish
convergence. Extended posterior prediction, density/mass/factor summaries and
calibration studies are separate research work; the current
[release work plan](../validation/release-plan.md) distinguishes those studies
from the tested workflow example.

```{literalinclude} ../../../scripts/example_classical_inference.py
:language: python
```
