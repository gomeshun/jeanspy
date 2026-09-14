# Units, shapes and model contracts

| Quantity | Unit / meaning |
| --- | --- |
| `R_pc`, `r_pc`, `x_pc`, `y_pc`, `re_pc`, `rs_pc`, `r_t_pc` | pc |
| `vmem_kms`, `vlos_kms`, `e_vlos_kms`, `sigmalos` | km/s |
| `sigmalos2`, intrinsic second moments | (km/s)² |
| `enclosed_mass` | solar masses |
| `rhos_Msunpc3`, halo `mass_density_3d` | solar masses / pc³ |
| Normalized tracer `density_3d`, `density_2d` | pc⁻³, pc⁻² |
| `inclination` | radians, 0 face-on, π/2 edge-on |
| `roi_deg` | degrees, circular cone half-angle |
| J, D factors | GeV² cm⁻⁵, GeV cm⁻² |

Spherical LOS solvers accept scalar or nonempty 1-D finite positive projected
radii. Classical LOS solvers return a scalar for scalar input and a matching
1-D array otherwise; JAX spherical LOS solvers always return a 1-D array
(length one for a scalar).
The classical backend raises for invalid radii; the JAX backend returns NaN
at invalid elements. Neither supports `R_pc=0`;
the central limit is model dependent. Axisymmetric coordinate pairs broadcast
to a common shape, include signed positions and the projected center, and
return that shape. Physical parameter dictionaries contain scalar values;
use `jax.vmap` to batch dictionaries explicitly. Inference data contain matching
nonempty finite 1-D arrays; measurement errors must be nonnegative.

Classical models are stateful: construct components, then update their named
parameters. JAX spherical models use explicit `params` dictionaries. The
axisymmetric forward configurations are frozen dataclasses in both backends;
use `dataclasses.replace` or construct another configuration to change node
counts. The convenience exports in `jeanspy.model` preserve the defining
axisymmetric classes.

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
