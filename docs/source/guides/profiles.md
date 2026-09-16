# Tracers, halos and anisotropy

This reference covers profile-specific scale, deprojection and cutoff
conventions. Use the [model tutorial](../tutorials/models.ipynb) to compose or
extend components and the [spherical API catalogue](../api/spherical.rst)
for available classes in each backend. Shared units and parameter interfaces
are defined in the [model contract](contracts.md).

## Tracer scales and three-dimensional support

The classical Plummer, projected exponential and Sersic profiles have
three-dimensional deprojections. `Uniform2dModel` supplies only a projected
disk and cannot be passed to a three-dimensional Jeans calculation.
`Exp3dModel` retains its historical scale convention: `re_pc` is an exponential
scale length, while `Exp2dModel.re_pc` is the projected half-light radius.

## Sersic deprojection domains

{meth}`~jeanspy.model.SersicModel.density_3d` selects the deprojection explicitly.
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
defines spherical and cylindrical anisotropy. The restriction $\beta<1$
alone does not establish the existence of a nonnegative distribution function.

## Halo cutoff conventions

Classical spherical halo density methods evaluate the untruncated profile;
their enclosed mass and factor calculations apply the configured cutoff.
The spheroidal `AxisymmetricZhaoModel.mass_density_3d` method instead returns zero outside its
ellipsoidal cutoff. Preserve these conventions when writing a custom integral.
