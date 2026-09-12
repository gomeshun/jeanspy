# Axisymmetric Jeans modeling and inference

JeansPy supports the stationary, cylindrically aligned, nonrotating model in
[Hayashi & Chiba (2015)](https://arxiv.org/abs/1507.07620), equations 1–5, and
[Hayashi et al. (2016)](https://arxiv.org/abs/1603.08046), equations 3–16.
Stellar and halo axes coincide, mixed velocity moments vanish, and
`beta_z = 1 - <vz²>/<vR²>` is constant. These assumptions define this extension
of [issue #52](https://github.com/gomeshun/jeanspy/issues/52).

| Capability | NumPy/SciPy | JAX/NumPyro |
| --- | --- | --- |
| Flattened Plummer tracer and oblate/prolate Zhao halo | Yes | Yes |
| Intrinsic moments, projected moments, densities, enclosed spheroidal mass | Yes | Yes, JIT and parameter autodiff |
| Optional finite spheroidal halo cutoff | Yes | Yes |
| Signed sky coordinates, inclination and photometric deprojection | Yes | Yes |
| Explicit priors and unbinned velocity likelihood | `AxisymmetricDSphEstimationModel` | `AxisymmetricJeansLikelihoodModel` and `ParameterSpec` |
| Posterior sampling and persisted restart checks | Existing `Sampler` / emcee | Existing `NumPyroSampler` / NUTS |
| J/D factors from posterior draws | Finite-cone quadrature | Postprocess draws with the same NumPy factor API |

The existing spherical classes retain their names and parameter meanings.
Cylindrical `beta_z` is **not** spherical `beta_ani`. In particular, the
spherical/isotropic comparison needs `q = Q = 1` and `beta_z = 0` together.
Spatially varying cylindrical anisotropy, arbitrary ellipsoid tilt, rotation
or Satoh decomposition, triaxiality, stellar self-gravity and PSF/bin averaging
are not implemented. The tracer family is Plummer; the halo family is Zhao.

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

```math
\rho(m)=\rho_s (m/r_s)^{-\gamma}
 \left[1+(m/r_s)^\alpha\right]^{(\gamma-\beta)/\alpha},
\qquad m^2=R^2+z^2/Q^2.
```

A finite `r_t_pc` sets this density to zero for m > r_t in the density, mass,
force and J/D calculations. `enclosed_mass(m_pc)` means mass inside the
**spheroid** m <= m_pc and includes the factor Q in its volume element. It
saturates outside r_t. The Hayashi 2015 halo uses `alpha=2, beta=3` and
`gamma=-alpha_paper`; NFW uses `alpha=1, beta=3, gamma=1`.

## Forward APIs

```python
import numpy as np
from jeanspy.model import AxisymmetricDSphModel

params = dict(re_pc=300., rs_pc=500., rhos_Msunpc3=.1,
              q_projected=.8, Q=.7, alpha=2., beta=3., gamma=.8,
              beta_z=-.3, inclination=np.deg2rad(70), r_t_pc=3000.)
model = AxisymmetricDSphModel(n_force=96, n_vertical=96, n_los=96)
radius = np.array([0., 100., 300., 900.])
major_sigma = np.sqrt(model.sigmalos2(radius, 0., params=params))
minor_sigma = np.sqrt(model.sigmalos2(0., radius, params=params))
vR2, vz2, vphi2 = model.intrinsic_moments(100., 50., params=params)
gR, gz = model.potential_gradient(100., 50., params=params)
```

The force method returns the **potential gradient**, opposite to gravitational
acceleration. Exact cusp-origin force evaluation is undefined in this API;
cored-origin force, intrinsic central moments and projected central moments
are supported. Intrinsic moments are ordered `(vR2, vz2, vphi2)`. These and LOS
outputs are second moments, equal to dispersion squared under the zero
streaming assumption used by the likelihood.

The component API from PR #63 also remains available:

```python
from jeanspy.axisymmetric import AxisymmetricJeans, PlummerTracer, ZhaoHalo

forward = AxisymmetricJeans(
    PlummerTracer(a_pc=300., q=.65),
    ZhaoHalo(rho_s=.1, r_s=500., Q=.7, r_t_pc=3000.),
    beta_z=-.3, inclination=1.1,
)
value = forward.los_second_moment(100., 50.)
```

Both forward configurations are immutable; use `dataclasses.replace` for
changes. NumPy raises `InvalidAxisymmetricModelError` (a `ValueError`) for
inadmissible physical parameters or negative/nonfinite intrinsic moments.

The corresponding JAX API accepts the same physical dictionary:

```python
# Configure JEANSPY_JAX_ENABLE_X64=true before starting Python for float64.
import jax
from jeanspy.model_numpyro import AxisymmetricDSphModel as JaxAxisymmetricModel

jax_model = JaxAxisymmetricModel(48, 48, 48)
value = jax_model.sigmalos2(100., 50., params=params)
derivative = jax.grad(lambda rho: jax_model.sigmalos2(
    100., 50., params={**params, "rhos_Msunpc3": rho}))(.1)
```

It uses JAX operations throughout, with no SciPy callbacks. Invalid dynamic
proposals give NaN forward outputs and are rejected by the likelihood.
Coordinate and schema shape errors raise immediately. Arrays broadcast and
scalar coordinates produce scalar outputs in both backends.

## Inference and restart

Observations require matching, finite, nonempty 1-D arrays `x_pc`, `y_pc`,
`vlos_kms`, `e_vlos_kms`. Errors must be nonnegative; zero is allowed. A
DataFrame or mapping is accepted by the classical model. The explicit
`AxisymmetricKinematicData` object copies observations and supplies detached
arrays with `as_kwargs()` for NumPyro calls. `reset_data` validates replacements
before mutation; user-provided priors are never derived from velocities.

The unbinned likelihood is

```math
\log L_i = \log\mathcal N\!\left(v_i\mid v_{\rm mem},
 \sqrt{\sigma_{\rm los}^2(x_i,y_i)+e_i^2}\right).
```

The classical prior table names the **sampling coordinates**, in sampler
order. Prefixes `log10_` and `bfunc_` mean `10**x` and `1-10**x`, respectively.
`cos_inclination` maps to `arccos(x)` in radians. Uniform bounds on this last
coordinate give an isotropic orientation prior over the explicitly chosen
range; use bounds compatible with photometry, or allow the deprojection
constraint to reject inadmissible proposals. No implicit transformation
Jacobian is added: the prior is defined in the named sampling coordinate.

```python
import pandas as pd
from jeanspy.axisymmetric_inference import AxisymmetricDSphEstimationModel
from jeanspy.sampler import Sampler

prior = pd.DataFrame({"lower": [-1.5, -30.], "upper": [-.5, 30.]},
                     index=["log10_rhos_Msunpc3", "vmem_kms"])
fixed = {name: value for name, value in params.items() if name != "rhos_Msunpc3"}
data = dict(x_pc=[30., -100., 300.], y_pc=[20., 70., -50.],
            vlos_kms=[1., -3., 5.], e_vlos_kms=[2., 2., 1.])
fit = AxisymmetricDSphEstimationModel(data, prior, fixed_params=fixed,
                                     dsph_model=model)
sampler = Sampler(fit, fit.sample, nwalkers=6, prefix="axisymmetric_")
# sampler.run_mcmc(iterations=100, loops=1)
```

An optional `PhotometryPriorModel` multiplies the prior on `log10_re_pc`;
initial points then use the appropriately truncated Gaussian. `fit.sample`
accepts an RNG/seed and bounded rejection to return feasible initial points.
Impossible support raises an error rather than changing the priors.
`lnlikelihoods`, `lnlikelihood`, `lnpriors`, `lnposterior`,
`lnposterior_wbic`, and `sample_data` are available. WBIC needs at least two
stars. Classical data are process-local and pickleable; shared-memory buffers
specific to the spherical implementation are not used.

NumPyro uses the existing `ParameterSpec` and sampler:

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jeanspy.sampler_numpyro import (
    AxisymmetricJeansLikelihoodModel, NumPyroSampler, ParameterSpec,
)

likelihood = AxisymmetricJeansLikelihoodModel(
    jax_model,
    [ParameterSpec.pow10("log10_rho", dist.Uniform(-1.5, -.5),
                          param_name="rhos_Msunpc3"),
     ParameterSpec("vmem_kms", dist.Normal(0., 30.))],
    fixed_params=fixed,
)
mcmc = MCMC(NUTS(likelihood), num_warmup=100, num_samples=100,
            num_chains=1, progress_bar=False)
# with NumPyroSampler(mcmc, output_dir="axisymmetric_nuts") as sampler:
#     sampler.run(jax.random.PRNGKey(52), **data)
```

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

Persisted analysis identity includes sky coordinates **in both axes**, velocity
errors, priors and transformations, parameter order, fixed physical parameters,
quadrature settings, source/dependency versions, and (for JAX) backend and
precision. A different analysis is rejected before appending samples. Restore
the original analysis or use a new output location. A missing NumPyro
`metadata.json` never gets silently regenerated for an existing chain.

The complete executable example samples scale radius, density, anisotropy,
inclination and systemic velocity, writes posterior draws and J/D factors,
and resumes when repeated with the same arguments:

```bash
python examples/axisymmetric_inference.py --backend classical --output-dir /tmp/axisym-emcee
JEANSPY_JAX_ENABLE_X64=true python examples/axisymmetric_inference.py \
    --backend numpyro --output-dir /tmp/axisym-nuts
```

Defaults use eight synthetic stars and short chains to exercise the workflow;
convergence, coverage and scientific calibration are not established by this
example. Increase warmup/draw counts and check diagnostics for an actual fit.
Use the same warmup setting when resuming the classical example, since its
stored warmup steps are explicitly discarded from the exported posterior.

## J and D factors

```python
J = model.jfactor(80000., .5, params=params)  # GeV^2 cm^-5
D = model.dfactor(80000., .5, params=params)  # GeV cm^-2
```

`roi_deg` is a **circular cone half-angle in degrees**, in [0,90). A finite
`r_t_pc` must be supplied. The observer must lie outside a sphere enclosing the
halo: `dist_pc > r_t_pc * max(1,Q)`. J needs `gamma < 1.5`; an integrated central
aperture with a steeper annihilation cusp diverges and raises. No central core
or artificial radius floor regularizes this divergence. D is finite over the
supported density-slope domain.

The implementation evaluates the exact observer integral

```math
F_p=\int_{\rm cone}\rho^p\,ds\,d\Omega
   =\int_{\rm halo\cap cone}\frac{\rho^p}{s^2}\,d^3r,
\qquad p=2\ (J),\quad p=1\ (D).
```

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

## Numerical equations and validation

Writing `P = nu*vz2`, the aligned Jeans equations are

```math
P(R,z)=\int_{|z|}^\infty \nu(R,z')\Phi_z(R,z')\,dz',\qquad
v_R^2=\frac{P}{(1-\beta_z)\nu},\qquad
\overline{v_\phi^2}=\frac{P+R P_R}{(1-\beta_z)\nu}+R\Phi_R.
```

The homoeoidal force has `D(t)²=1+(Q²-1)t²`,
`m(t)²=t²*(R²+z²/D(t)²)`, and

```math
\Phi_R=4\pi GQR\int_0^{t_{\max}}\frac{\rho(m)t^2}{D(t)}\,dt,
\qquad
\Phi_z=4\pi GQz\int_0^{t_{\max}}\frac{\rho(m)t^2}{D(t)^3}\,dt.
```

Without a cutoff, t_max=1. Outside a finite halo, m(t_max)=r_t.
The analytic radial derivative includes the moving endpoint contribution
`integrand(t_max)*dt_max/dR`; omitting it would give incorrect azimuthal
moments outside the cutoff. The pressure boundary condition is P -> 0 at
infinity, and stellar tracers continue outside the finite halo.

Vertical and LOS integrals use fixed Gauss–Legendre nodes with rational
infinity maps. LOS quadrature is centered at maximum tracer density. Defaults
are 96 nodes per integral; numerical settings must be refined for the actual
parameter regime, especially sharp truncation, extreme scale ratios/cusps,
flattening and outer positions. There is no global adaptive accuracy guarantee.
Cost grows approximately as `N_stars*n_force*n_vertical*n_los`. NumPy processes
one sky position at a time; JAX uses `lax.map` and rematerialization to bound
intermediate memory during parameter differentiation.

The recorded [CPU runtime sample](../validation/axisymmetric_runtime.json)
uses eight stars and float64. At 96 nodes per integral, one NumPy prediction
took 1.54 s, a warm JAX prediction 0.105 s, and a warm gradient of the summed
predictions with respect to all physical parameters 0.476 s. First calls,
including compilation, are recorded separately. These are single-run timings
for the stated parameters and machine, not an inference-throughput or accuracy
guarantee. Reproduce the measurement with:

```bash
JEANSPY_JAX_ENABLE_X64=true python scripts/benchmark_axisymmetric.py \
    --output validation/axisymmetric_runtime.json
```

Automated checks cover:

- Exact spherical Plummer intrinsic/LOS moments; existing spherical NFW Jeans
  and force limits, including finite truncation.
- Oblate/prolate Plummer projection and geometry; force parity, Poisson and
  Jeans equations, the cutoff derivative, and quadrature refinement.
- An independent flattened, face-on Plummer-tracer reference with analytic
  Plummer force and adaptive LOS integration.
- Matched NumPy/JAX values, all physical-parameter derivatives, finite-cutoff
  derivatives, cored-origin force gradients, invalid proposals, and float32.
- Explicit prior schemas, physical rejection, data replacement, emcee export,
  real NUTS/emcee restart, and rejection before changing stored samples.
- Exact finite-cone J/D geometry against independent observer-ray quadrature;
  the spherical J limit against `jfactor_ullio2016`, convergence and divergent
  central-cusp rejection.

The independent [JAM benchmark](../validation/axisymmetric_jam_reference.json)
records two oblate/prolate Zhao cases, all MGE coefficients, package versions,
a kernel SHA256, effective calculation paths and numerical refinements.
Reproduce it with the stored MGE coefficients:

```bash
python scripts/validate_axisymmetric_jam.py \
    --reuse-mge validation/axisymmetric_jam_reference.json \
    --output /tmp/axisymmetric_jam_recheck.json
```

Omit `--reuse-mge` to refit the MGE approximations with requested Gaussian
budgets of 32 and 48. Nonnegative fitting can retain fewer components; the
report records both the requested and retained counts. It stores both sets
of coefficients so an integration recheck can hold the approximation fixed.

It uses Cappellari (2008) equation (28) as implemented by jampy, independently
of JeansPy's three nested integrations. The JAM gravitational constant is
explicitly converted to JeansPy's constant. In the recorded run the maximum
JeansPy/reference differences are 1.13e-5 and 2.87e-5 in second moments; MGE
order refinement changes the reference by at most 7.75e-4. The declared
agreement gate is 3e-3, accounting for the MGE approximation.

In jampy 8.1.4, `interp=False` at the public entry point overrides
`analytic_los=True` and selects numerical LOS integration with internally
interpolated intrinsic moments. The analytic benchmark therefore requests
`interp=True` and checks that the returned `vel2` tensor is `None`, confirming
the analytic path. With these ten positions, no PSF or pixel averaging and
`nrad*nang > 10`, that path evaluates each requested position directly rather
than interpolating an output grid.

The public analytic path agrees with the independently integrated kernel to
7.39e-9 and 1.75e-10 in relative second moments. Direct calls to JAM's standard
`quad1d` and SciPy quadrature with explicit breakpoints in u confirm the
log(u) integral. There is no evidence in these cases that JAM's standard
one-dimensional kernel quadrature fails to resolve the MGE.

The report also retains the `interp=False` numerical-LOS results. At the
default 20-by-10 intrinsic grid, their maximum differences are 1.27% and
2.46%. Increasing only that grid to 40-by-20 reduces them to 0.233% and
0.441%; using an 80-by-40 grid, 3000 LOS points and `epsrel=1e-6` reduces
them to 0.0596% and 0.115%. This establishes resolution dependence in the
numerical path, not a discrepancy in the analytic JAM equation. Those
finite-grid residuals are not global error bounds.

The two cases and ten positions do not establish an accuracy guarantee over
all parameters. MGE approximation and JeansPy quadrature still need their own
refinement checks. These JAM cases have no halo cutoff; finite-cutoff checks
are covered by separate tests. The benchmark does not reproduce the Hayashi
galaxy fits or establish coverage/calibration of a real-data inference.

The [workflow validation record](../validation/axisymmetric_workflow.md)
collects the integration-test results, persisted example runs and installed
wheel/source-distribution checks.
