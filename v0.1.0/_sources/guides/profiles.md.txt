# Tracers, halos and anisotropy

This reference covers profile-specific scale, deprojection and cutoff
conventions. Use the [model tutorial](../tutorials/models.ipynb) to compose or
extend components and the [spherical API catalogue](../api/spherical.rst)
for available classes in each backend. Shared units and parameter interfaces
are defined in the [model contract](contracts.md).

## Tracer scales and three-dimensional support

The NumPy/SciPy Plummer, projected exponential and Sersic profiles have
three-dimensional deprojections. `Uniform2dModel` supplies only a projected
disk and cannot be passed to a three-dimensional Jeans calculation.
`ProjectedExponentialModel(r_exp_pc=...)` uses the scale inside
`exp(-R/r_exp_pc)`. Its read-only `re_pc` property returns the projected
half-light radius, `1.67834699001666 * r_exp_pc`. This profile is a projected
exponential with a Bessel-K0 three-dimensional deprojection, not a pure
three-dimensional exponential. Plummer and Sersic still accept `re_pc` directly.

## Sersic deprojection domains

{meth}`~jeanspy.model.SersicModel.density_3d` selects the deprojection explicitly.
The `lgm` selection uses the Lima Neto--Gerbal--Márquez approximation for
$0.5\le n\le10$; its normalization is `lgm_norm_3d`, independently of `auto`.
The VM20 approximation is restricted to $0.5\le n\le10$ and
$10^{-3}\le r/r_e\le10^3$; VM20bis uses $0.5\le n\le3.4$ and
$10^{-4}\le r/r_e\le10^3$. The default `auto` route selects VM20bis inside
its domain, the SP04 approximation for $3.4<n\le10$ over that radius range,
and numerical Abel integration outside these domains. Explicit approximation
VM20/VM20bis methods reject unsupported values instead of extrapolating them silently.
See that method's API entry for the quadrature controls of `numerical`.

## Anisotropy families

The spherical implementations include
{class}`~jeanspy.model.ConstantAnisotropyModel`,
{class}`~jeanspy.model.OsipkovMerrittModel` and
{class}`~jeanspy.model.BaesAnisotropyModel`; consult their API entries for
parameters and supported kernels. The [theory page](../theory.md)
defines spherical and cylindrical anisotropy.
For JAX `BaesEta2AnisotropyModel`, `solver="auto"` uses Abel integration.
Select `solver="kernel"` to activate its specialized Appell-F1 kernel, controlled
by `n_kernel`. The eta=2 specialization alone does not change solver selection. The restriction $\beta<1$
alone does not establish the existence of a nonnegative distribution function.

## Halo cutoff conventions

Spherical NFW and Zhao `mass_density_3d` methods return zero for `r > r_t_pc`
in both NumPy/SciPy and JAX; the boundary is included. Their enclosed mass is
constant outside the same cutoff. `r_t_pc=np.inf` gives an untruncated halo
at finite radii. Zhao uses `alpha`, `beta`, `gamma` for transition, outer and
inner slopes in both geometries.

The spheroidal `AxisymmetricZhaoModel` applies this convention to
`m = sqrt(R**2 + z**2/Q**2)`. Its enclosed mass is inside that ellipsoid.
For annihilation factors, use the [factor guide](factors.md) to select the
finite cone or an explicitly named approximation.
