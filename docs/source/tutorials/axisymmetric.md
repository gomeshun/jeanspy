# Axisymmetric synthetic analysis

This example samples scale radius, density, cylindrical anisotropy,
inclination and systemic velocity, and computes finite-cone J/D factors from
a deterministic subset of saved posterior draws. Priors are explicit in
`log10_rhos_Msunpc3`, `log10_rs_pc`, `bfunc_beta_z`, `cos_inclination` and
`vmem_kms`.

```bash
python examples/axisymmetric_inference.py \
    --backend classical --output-dir /tmp/axisymmetric-emcee
JEANSPY_JAX_ENABLE_X64=true python examples/axisymmetric_inference.py \
    --backend numpyro --output-dir /tmp/axisymmetric-nuts
```

Repeat the same command to resume. Keep the same warmup setting when resuming
the classical example so that its export discards the original warmup steps.
Outputs include `observations.csv`, `prior.csv`, `posterior.csv`,
`derived_factors.csv`, `summary.json` and persisted sampler state.

Default settings use eight stars and short chains. They exercise the complete
storage path; the summary explicitly records that convergence and calibration
are unestablished. The [release work plan](../validation/release-plan.md)
tracks the expanded analysis and posterior predictive diagnostics.

```{literalinclude} ../../../examples/axisymmetric_inference.py
:language: python
```
