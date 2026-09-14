# Tracers, halos and anisotropy

Classical components store scalar physical parameters. Update them with
`update` before evaluating a new model. Densities use plain numbers in pc,
solar masses and km/s. A normalized stellar density has unit total tracer
number; it contributes no stellar mass to the gravitational potential.

```{literalinclude} ../../../examples/docs_profiles.py
:language: python
:end-before: profiles-end
```

The classical Plummer, projected exponential and Sersic profiles have
three-dimensional deprojections. `Uniform2dModel` supplies only a projected
disk and cannot be passed to a three-dimensional Jeans calculation.
`Exp3dModel` retains its historical scale convention: `re_pc` is an exponential
scale length, while `Exp2dModel.re_pc` is the projected half-light radius.

`SersicModel.density_3d(..., method=...)` selects the deprojection explicitly.
The VM20 approximation is restricted to $0.5\le n\le10$ and
$10^{-3}\le r/r_e\le10^3$; VM20bis uses $0.5\le n\le3.4$ and
$10^{-4}\le r/r_e\le10^3$. The default `auto` route selects VM20bis inside
its domain, the SP04 approximation for $3.4<n\le10$ over that radius range,
and numerical Abel integration outside these domains. Explicit approximation
VM20/VM20bis methods reject unsupported values instead of extrapolating them silently.
See the API methods for the quadrature controls of `numerical`.

Spherical anisotropy is $\beta=1-\sigma_\theta^2/\sigma_r^2$ for one tangential
component. It is distinct from cylindrical $\beta_z$ in the axisymmetric
solver. The restriction $\beta<1$ alone does not establish the existence of
a nonnegative distribution function.

```{literalinclude} ../../../examples/docs_profiles.py
:language: python
:start-after: anisotropy-start
:end-before: anisotropy-end
```

Classical spherical halo density methods evaluate the untruncated profile;
their enclosed mass and factor calculations apply the configured cutoff.
The spheroidal `ZhaoHalo.density` method instead returns zero outside its
ellipsoidal cutoff. Preserve these conventions when writing a custom integral.
