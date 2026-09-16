# J and D factors

For a density $\rho$, annihilation and decay factors are

```{math}
J=\int_{\Delta\Omega}\int\rho^2\,ds\,d\Omega,\qquad
D=\int_{\Delta\Omega}\int\rho\,ds\,d\Omega.
```

These are NumPy postprocessing operations. Use physical posterior draws from
either sampler; keep the same distance, aperture, cutoff and density convention
when comparing results. The returned units are GeV² cm⁻⁵ and GeV cm⁻².

## Spherical apertures

The spherical `DMModel.jfactor_ullio2016` method includes outer halo shells
projected into a finite cone. Its `simple` variant uses a spherical-aperture
approximation. The two are different integrals. The
[geometry derivation](ullio-geometry.md) follows
[Ullio & Valli (2016)](https://arxiv.org/abs/1603.07721).

The spherical limit of `AxisymmetricZhaoModel` also supplies a D-factor
calculation for a spherical halo; there is no separate classical spherical
`DMModel.dfactor` API.

```{literalinclude} ../../../examples/docs_factors.py
:language: python
:end-before: spherical-factors-end
```

## Axisymmetric finite-cone factors

`roi_deg` is a **circular cone half-angle in degrees**, in [0,90). A finite
`r_t_pc` must be supplied. The observer must lie outside a sphere enclosing the
halo: `dist_pc > r_t_pc * max(1,Q)`. J needs `gamma < 1.5`; an integrated central
aperture with a steeper annihilation cusp diverges and raises. No central core
or artificial radius floor regularizes this divergence. D is finite over the
supported density-slope domain. See the [axisymmetric guide](axisymmetric.md)
for the physical parameters and viewing geometry.

The implementation evaluates the exact observer integral

$$
F_p=\int_{\rm cone}\rho^p\,ds\,d\Omega
   =\int_{\rm halo\cap cone}\frac{\rho^p}{s^2}\,d^3r,
\qquad p=2\ (J),\quad p=1\ (D).
$$

Spheroidal volume coordinates have `d³r = Q m² dm dmu dphi`. For a unit
spheroidal direction with LOS component L and sky-plane length B, the cone
bounds m by `m*(B*cos(theta)-L*sin(theta)) <= dist_pc*sin(theta)` and the
observer distance is `s² = dist_pc² + 2*dist_pc*m*L + m²*|e|²`.
This retains finite-distance geometry and contributions from outer shells
projected into the aperture. It does not replace the aperture by a sphere.

Radial quadrature is cusp-regularized analytically and split at rs; angles use
Gauss–Legendre and periodic azimuth quadrature. Expose accuracy with `n_mu`,
`n_phi`, `n_radial` (defaults 96,96,128). The cone/cutoff intersection can
require angular refinement. These settings are independent of the three Jeans
quadrature orders. J/D evaluation is a NumPy postprocessing operation for
chains from either backend, as in the spherical workflow.

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
