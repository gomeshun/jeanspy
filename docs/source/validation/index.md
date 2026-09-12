# Validation and reproducibility

Numerical tests, sampler convergence and scientific calibration answer
different questions. Passing tests checks the implemented contracts. Agreement
with an independent integral checks selected numerical cases. Recovery and
coverage experiments assess statistical performance under specified data
generating assumptions.

Existing evidence includes spherical numerical regression tests, NumPy/JAX
consistency, axisymmetric spherical limits and Jeans/Poisson identities,
finite-difference parameter gradients, observer-ray J/D integration and a
JAM comparison with recorded MGE coefficients. The
[axisymmetric guide](../guides/axisymmetric.md) links the original reports and
reproduction commands.

Earlier short-chain comparisons are preliminary evidence. Their raw-star
scaling used a fixed number of dispersion bins and does not measure the cost
of an unbinned likelihood as the number of stars grows. Their posterior
medians and acceptance fractions are insufficient to establish converged
posterior agreement or coverage.

```{toctree}
:maxdepth: 1

release-plan
jam9
```
