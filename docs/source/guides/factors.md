# J and D factors

For a density $\rho$, annihilation and decay factors are

```{math}
J=\int_{\Delta\Omega}\int\rho^2\,ds\,d\Omega,\qquad
D=\int_{\Delta\Omega}\int\rho\,ds\,d\Omega.
```

These are NumPy postprocessing operations. Use physical posterior draws from
either sampler; keep the same distance, aperture, cutoff and density convention
when comparing results. The returned units are GeV² cm⁻⁵ and GeV cm⁻².

The spherical `DMModel.jfactor_ullio2016` method includes outer halo shells
projected into a finite cone. Its `simple` variant uses a spherical-aperture
approximation. The two are different integrals. The
[geometry derivation](ullio-geometry.md) follows
[Ullio & Valli (2016)](https://arxiv.org/abs/1603.07721).

Axisymmetric `jfactor` and `dfactor` require finite `r_t_pc` and an observer
outside a sphere enclosing the halo: `dist_pc > r_t_pc * max(1, Q)`.
The circular aperture half-angle lies in `[0, 90)` degrees. J requires
`gamma < 1.5`; a steeper central cusp diverges, and no artificial core or radius
floor is introduced. Angular and radial factor quadrature counts are
independent of the Jeans solver's node counts. Refine them for the actual
cutoff/cone intersection.

The spherical limit of `ZhaoHalo` also supplies a D-factor calculation for a
spherical halo; there is no separate classical spherical `DMModel.dfactor` API.

```{literalinclude} ../../../examples/docs_factors.py
:language: python
:end-before: spherical-factors-end
```

For a flattened halo the factor depends on viewing inclination:

```{literalinclude} ../../../examples/docs_factors.py
:language: python
:start-after: flattened-factors-start
:end-before: flattened-factors-end
```

```{toctree}
:hidden:

ullio-geometry
```
