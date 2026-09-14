# Inference, diagnostics and restart

Specify the data, physical model, sampling coordinates, priors, fixed parameters
and numerical settings explicitly. The classical `FlatPriorModel` uses an
ordered table with finite `lower` and `upper` bounds. Spherical convenience
models also use an explicit photometric prior on `log10_re_pc`. A prior on
systemic velocity is not inferred from the observed velocities unless the
caller explicitly requests the data-derived option.

NumPyro uses `ParameterSpec` objects with distributions, optional transforms
and physical parameter names. Sampled and fixed physical parameters must not
overlap. The provided `JeansLikelihoodModel` and
`AxisymmetricJeansLikelihoodModel` reject inadmissible forward variances.
Choose the same likelihood and the same prior measure for sampler comparisons.

`fixed_params` is an argument of `AxisymmetricJeansLikelihoodModel`. The
spherical `JeansLikelihoodModel` instead allows a `parameter_postprocess`
callable to assemble a physical parameter dictionary from sampled values.
Read the actual constructor signature for the selected geometry.

This classical example writes and resumes six steps in a temporary directory:

Its autocorrelation estimate may be undefined because the chain is deliberately
short. The wrapper still computes that diagnostic when early stopping is
disabled; neither its presence nor its printed value establishes convergence.

```{literalinclude} ../../../examples/docs_inference.py
:language: python
:end-before: classical-inference-end
```

This NumPyro example writes two chunks and reconstructs the sampler between
them. Six warmup steps and eight saved draws verify the storage workflow only:

```{literalinclude} ../../../examples/docs_numpyro_inference.py
:language: python
:end-before: numpyro-inference-end
```

Use multiple independent chains, rank-normalized R-hat, bulk and tail ESS,
Monte Carlo uncertainty, trace plots and posterior predictive checks.
Inspect NUTS divergences and tree depth as well as emcee autocorrelation and
acceptance. Emcee walkers interact; do not treat individual walkers as
independent chains for R-hat. A short saved chain is a workflow example, not
evidence of convergence, coverage or scientific calibration.

`Sampler` persists emcee chains in HDF5. `NumPyroSampler` supports the declared
ArviZ storage backends and writes analysis identity metadata. Reconstruct the
same analysis and point to the existing output to resume. Identity includes
data, priors, model configuration, source/dependencies and, for NumPyro, the
effective JAX backend and precision. Changing these requires a new output.
If `metadata.json` is missing for an existing NumPyro analysis, restore the
original metadata or use a new directory; do not create a replacement identity
from the current inputs.

The [tutorials](../tutorials/index.md) provide executable save/resume commands.
Keep raw draws and diagnostics even when a stopping criterion fails. A smaller
time per effective sample is useful computational evidence only after its
estimation assumptions hold; it is not evidence of calibrated uncertainty.
