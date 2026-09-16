# Axisymmetric Jeans modeling and inference

Use this guide for geometry, physical parameter domains and the differences
from spherical models. For a complete executable workflow, follow the
[axisymmetric synthetic analysis](../tutorials/axisymmetric.md). The
[axisymmetric API reference](../api/axisymmetric-models.rst) documents
constructors and individual methods; [Theory](../theory.md#axisymmetric-jeans-equations)
gives the dynamical assumptions and equations.
For factor integrals and numerical settings, see the
[J/D-factor guide](factors.md#axisymmetric-finite-cone-factors) and
[quadrature reference](numerics.md#axisymmetric-quadrature).

The supported components are a flattened Plummer tracer, an oblate or prolate
Zhao halo and constant cylindrical anisotropy. Cylindrical `beta_z` is distinct
from spherical `beta_ani`; the common spherical/isotropic limit requires
`q = Q = 1` and `beta_z = 0` together.

## Units, geometry and physical parameters

Distances are pc, masses Msun, velocities km/s. `inclination` is in **radians**:
zero is face-on and pi/2 is edge-on. Sky `x_pc` is the line of nodes, the
projected major axis for an oblate tracer, and `y_pc` is perpendicular to it.
For a prolate tracer x need not be the projected major axis. Rotate an observed
catalog by its position angle and convert angular separations to pc before
passing it to this API. Signed coordinates and the projected center are valid.

| Parameter | Meaning/domain |
| --- | --- |
| `re_pc` | Positive equatorial Plummer scale; projected major-axis half-light radius for an oblate tracer |
| `rs_pc`, `rhos_Msunpc3` | Positive Zhao scale radius and density scale; rho_s is not rho(rs) |
| `q` | Positive intrinsic stellar vertical/equatorial ratio |
| `q_projected` | Alternative to q: oblate projected ratio in (0,1], deprojected using inclination |
| `Q` | Positive intrinsic halo vertical/equatorial ratio; default 1 |
| `alpha`, `beta`, `gamma` | Zhao exponents: alpha > 0, beta > 2, 0 <= gamma < 2; defaults 1,3,1 |
| `beta_z` | Finite and < 1; default 0; positive moments impose additional constraints |
| `inclination` | In [0,pi/2]; default pi/2 |
| `r_t_pc` | Optional positive equatorial ellipsoidal cutoff; default infinity |
| `vmem_kms` | Systemic velocity, explicitly fixed or sampled for inference |

Supply exactly one of `q` and `q_projected`. Oblate photometry obeys
`q_projected² = cos(i)² + q² sin(i)²`; deprojection requires nonzero inclination
and positive intrinsic q². `intrinsic_axis_ratio(q_projected, inclination)`
provides the same conversion. A round image at exactly zero inclination is
degenerate and is rejected; use an explicitly supplied intrinsic q for a
face-on model. Unknown parameter names and nonscalar physical parameters are
errors. Use `jax.vmap` over parameter dictionaries for batched JAX predictions.

The halo density is

$$
\rho(m)=\rho_s (m/r_s)^{-\gamma}
 \left[1+(m/r_s)^\alpha\right]^{(\gamma-\beta)/\alpha},
\qquad m^2=R^2+z^2/Q^2.
$$

A finite `r_t_pc` sets this density to zero for m > r_t in the density, mass,
force and J/D calculations. `enclosed_mass(m_pc)` means mass inside the
**spheroid** m <= m_pc and includes the factor Q in its volume element. It
saturates outside r_t. The Hayashi 2015 halo uses `alpha=2, beta=3` and
`gamma=-alpha_paper`; NFW uses `alpha=1, beta=3, gamma=1`.

## Forward APIs

Both implementations compose `AxisymmetricPlummerModel`,
`AxisymmetricZhaoModel` and `AxisymmetricConstantAnisotropyModel` inside
`AxisymmetricDSphModel`. Choose the imports and parameter interface below:

| Implementation | Import module | Physical parameters |
| --- | --- | --- |
| NumPy/SciPy | `jeanspy.model` | Stored in immutable components; optional per-call `params` overrides |
| JAX | `jeanspy.model_numpyro` | Passed explicitly at evaluation time |

The component interfaces are `AxisymmetricStellarModel`, `AxisymmetricDMModel`
and `AxisymmetricAnisotropyModel`. Zhao exponents retain the names
`alpha`, `beta`, `gamma` in this geometry.

The force method returns the **potential gradient**, opposite to gravitational
acceleration. Exact cusp-origin force evaluation is undefined in this API;
cored-origin force, intrinsic central moments and projected central moments
are supported. Intrinsic moments are ordered `(vR2, vz2, vphi2)`. These and LOS
outputs are second moments, equal to dispersion squared under the zero
streaming assumption used by the likelihood.

The components and forward configuration are immutable; use
`dataclasses.replace` to construct changed components or quadrature settings.
A per-call `params` dictionary overrides stored values without changing the
model. `model.physical_params` returns a detached dictionary of all stored
physical values. Supplying `q_projected` as an override replaces the stored
intrinsic `q` and applies the same deprojection rule.

NumPy raises `InvalidAxisymmetricModelError` (a `ValueError`) for inadmissible
physical parameters or negative/nonfinite intrinsic moments.

JAX uses JAX operations throughout, with no SciPy callbacks. Invalid dynamic
proposals give NaN forward outputs and are rejected by the likelihood.
Coordinate and schema shape errors raise immediately. Arrays broadcast and
scalar coordinates produce scalar outputs in both backends. For JIT setup,
runtime precision and gradient checks, use the [backend tutorial](../tutorials/backends.ipynb).

## Inference and restart

Observations require matching, finite, nonempty 1-D arrays `x_pc`, `y_pc`,
`vlos_kms`, `e_vlos_kms`. Errors must be nonnegative; zero is allowed. A
DataFrame or mapping is accepted by the classical model. The explicit
`AxisymmetricKinematicData` object copies observations and supplies detached
arrays with `as_kwargs()` for NumPyro calls. `reset_data` validates replacements
before mutation; user-provided priors are never derived from velocities.

The [likelihood equation](../theory.md#likelihood-and-interpretation) and
[MCMC tutorial](../tutorials/inference.ipynb) explain the statistical model.
The classical wrapper is
{class}`~jeanspy.axisymmetric_inference.AxisymmetricDSphEstimationModel`;
NumPyro uses {class}`~jeanspy.sampler_numpyro.AxisymmetricJeansLikelihoodModel`.

The classical prior table names the **sampling coordinates**, in sampler
order. Prefixes `log10_` and `bfunc_` mean `10**x` and `1-10**x`, respectively.
`cos_inclination` maps to `arccos(x)` in radians. Uniform bounds on this last
coordinate give an isotropic orientation prior over the explicitly chosen
range; use bounds compatible with photometry, or allow the deprojection
constraint to reject inadmissible proposals. No implicit transformation
Jacobian is added: the prior is defined in the named sampling coordinate.

An optional `PhotometryPriorModel` multiplies the prior on `log10_re_pc`;
initial points then use the appropriately truncated Gaussian. `fit.sample`
accepts an RNG/seed and bounded rejection to return feasible initial points.
Impossible support raises an error rather than changing the priors.
`lnlikelihoods`, `lnlikelihood`, `lnpriors`, `lnposterior`,
`lnposterior_wbic`, and `sample_data` are available. WBIC needs at least two
stars. Classical data are process-local and pickleable; shared-memory buffers
specific to the spherical implementation are not used.

`ParameterSpec` supports general priors/transforms and deterministic physical
sites. Sampled and fixed physical names must be disjoint; optional
postprocessing receives their combined dictionary. The NumPyro likelihood
also accepts the existing configurable velocity-mean and observation-distribution
hooks. Without postprocessing, construction checks required and supported
physical names, exactly one of `q` and `q_projected`, and the parameter selected
by a named `velocity_mean` (default `vmem_kms`). A callable mean does not require
`vmem_kms`. With postprocessing, these checks apply to its output immediately
before evaluating the forward model; construction does not execute callbacks or
sample priors. Configuration errors raise `ValueError`, while inadmissible
physical proposals still receive log probability minus infinity.

Its `sigma2_bounds` are rejection limits (default 1e-12 to 1e12 in
(km/s)²); finite variances outside them are rejected, never clipped into range.

Both inference paths reject inadmissible Jeans moments with log probability
minus infinity. This is not a proof of a positive distribution function or of
moment positivity at unobserved positions.

The [save/resume tutorial](../tutorials/storage.ipynb) describes persisted
analysis identity. For this geometry it includes sky coordinates **in both
axes**, priors and transformations, parameter order, fixed physical values and
quadrature settings, in addition to the shared source/dependency and JAX
runtime checks. Follow the [worked analysis](../tutorials/axisymmetric.md#3-save-resume-and-inspect)
for executable restart commands and output files.
