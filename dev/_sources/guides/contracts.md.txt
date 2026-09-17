# Units, shapes and model contracts

Use this page as a cross-backend reference. The
[units tutorial](../tutorials/units.ipynb) walks through preparing observations;
the [model tutorial](../tutorials/models.ipynb) demonstrates parameter updates.

| Quantity | Unit / meaning |
| --- | --- |
| `R_pc`, `r_pc`, `x_pc`, `y_pc`, `re_pc`, `r_exp_pc`, `rs_pc`, `r_t_pc` | pc |
| `vmem_kms`, `vlos_kms`, `e_vlos_kms`, `sigmalos` | km/s |
| `sigmalos2`, intrinsic second moments | (km/s)² |
| `enclosed_mass` | solar masses |
| `rhos_Msunpc3`, halo `mass_density_3d` | solar masses / pc³ |
| Normalized tracer `density_3d`, `density_2d` | pc⁻³, pc⁻² |
| `inclination` | radians, 0 face-on, π/2 edge-on |
| `roi_deg` | degrees, circular cone half-angle |
| J, D factors | GeV² cm⁻⁵, GeV cm⁻² |

Spherical LOS solvers accept scalar or nonempty 1-D finite positive projected
radii. NumPy/SciPy LOS solvers return a scalar for scalar input and a matching
1-D array otherwise; JAX spherical LOS solvers always return a 1-D array
(length one for a scalar).
The NumPy/SciPy backend raises for invalid radii; the JAX backend returns NaN
at invalid elements. Neither supports `R_pc=0`;
the central limit is model dependent. Axisymmetric coordinate pairs broadcast
to a common shape, include signed positions and the projected center, and
return that shape. Physical parameter dictionaries contain scalar values;
use `jax.vmap` to batch dictionaries explicitly. Inference data contain matching
nonempty finite 1-D arrays; measurement errors must be nonnegative.

NumPy/SciPy spherical models store parameters and support `update`. Axisymmetric
NumPy components also store parameters, but are immutable: use
`dataclasses.replace` to change a component or the forward model's node counts.
An axisymmetric per-call `params` mapping overrides stored physical values
without modifying the components. Both JAX geometries receive physical
parameters explicitly. The convenience exports in `jeanspy.model` preserve
the defining axisymmetric classes.

NumPy raises `ValueError` (including `InvalidAxisymmetricModelError`) for invalid
axisymmetric models. JAX uses NaN forward values for invalid dynamic physical
proposals, which the likelihood rejects with minus infinity; schema and shape
errors still raise before evaluation. Automatic differentiation is valid only
where the chosen numerical expression is differentiable and the model is
admissible. Boundaries, density cutoffs, rejection masks, discrete options and
NumPy postprocessing require separate treatment.

Spherical `beta_ani` and cylindrical `beta_z` describe different tensors.
The common spherical/isotropic limit requires `q=Q=1` and `beta_z=0`.
Axisymmetric mass is inside an ellipsoid, including its axis-ratio volume
factor. It is not a spherical enclosed mass except when `Q=1`.

## Observation precision and sampled parameters

`SimpleDSphEstimationModel(dtype=None)` stores observations in the common
floating dtype of the three input columns; integer-only data use float64.
Pass `dtype=np.float32` or `dtype=np.float64` to choose explicitly. Shared
buffers retain this dtype and reject a reset requiring a different dtype or
shape. The numerical solver may promote arithmetic precision. JAX precision
is configured separately in the [backend tutorial](../tutorials/backends.ipynb).

NumPy/SciPy inference accepts ordered `SamplingParameter(sample_name,
param_name, transform)` objects from `jeanspy.parameters`. Supported transforms
are `identity`, `pow10`, `one_minus_pow10`, and `arccos`. Without specifications,
names map by identity only. NumPyro uses distribution-bearing `ParameterSpec`
objects from `jeanspy.sampler_numpyro`. In both cases the prior is defined in
the sampled coordinate, with no implicit transformation selected by its name.
See the [API migration guide](api-migration.md) for complete mappings.
