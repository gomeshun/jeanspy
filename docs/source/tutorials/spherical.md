# Spherical synthetic analysis

The existing base-package example generates velocities from a Plummer tracer
and NFW halo, records explicit priors, samples with emcee, saves HDF5 draws and
checks that reopening and appending preserves the original chain.

Pass `model.lnposterior` to emcee, or a process pool to
`Sampler(..., pool=pool)`. Parallel emcee requires a
[picklable probability callable](https://emcee.readthedocs.io/en/v3.1.6/tutorials/parallel/#pickling-data-transfer-arguments);
scripts using a `spawn` pool must also guard process creation with
`if __name__ == "__main__":`.

```bash
python scripts/example_classical_inference.py --output-dir /tmp/spherical-example
```

Use a new output directory. Inputs and outputs include `observations.csv`,
`prior.csv`, the persisted sampler state and `result.json`. The seed is 55.
The displayed run contains only 12 stored steps and does not establish
convergence. Longer runs and posterior diagnostics are needed for inference.

```{literalinclude} ../../../scripts/example_classical_inference.py
:language: python
```
