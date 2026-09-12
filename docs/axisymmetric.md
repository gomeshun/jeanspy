# Axisymmetric Jeans models

Implementation branch: `feat/axisymmetric-jeans-hayashi`.

## Reference and scope

Hayashi & Chiba (2015), *Structural properties of non-spherical dark halos
in Milky Way and Andromeda dwarf spheroidal galaxies*, ApJ 810, 22:
https://arxiv.org/abs/1507.07620 (equations 1–5, section 3.1–3.2).
The paper PDF was checked directly during implementation.

The new, separate NumPy/SciPy API will solve the steady, cylindrically aligned
Jeans equations with zero mixed moments, a constant
`beta_z = 1 - <vz²>/<vR²>`, and an isolated boundary condition
`nu <vz²> -> 0` at infinity. Stellar and halo symmetry axes coincide.
The output is a second moment, not a decomposition into rotation and dispersion.
It equals the velocity dispersion squared only when mean streaming is zero.

Planned checkpoints:
1. Document equations, conventions and numerical validation targets.
2. Add flattened Plummer tracers, spheroidal Zhao halos, intrinsic and LOS moments.
3. Verify analytic spherical limits, flattened forces, Jeans residuals,
   inclination geometry and quadrature convergence; add a runnable example.

Distances are pc, masses solar masses, velocities km/s, angles radians.
Inclination zero is face-on; pi/2 is edge-on. Sky x is along the line of nodes.
Halo Q and stellar q are intrinsic vertical/equatorial axis ratios.
The spherical comparison requires q=Q=1 AND beta_z=0: cylindrical anisotropy
is not the spherical Jeans anisotropy parameter.

This is a forward solver. JAX differentiation, sampling integration, rotation
prescriptions, misaligned/triaxial systems, PSF/bin averaging and fitting the
paper's observed galaxies are outside this initial implementation.

## Implemented API

```python
import numpy as np
from jeanspy.axisymmetric import AxisymmetricJeans, PlummerTracer, ZhaoHalo

model = AxisymmetricJeans(
    tracer=PlummerTracer(a_pc=300, q=0.65),
    halo=ZhaoHalo(rho_s=0.1, r_s=500, Q=0.7, alpha=2, beta=3, gamma=1),
    beta_z=-0.2,
    inclination=np.deg2rad(70),
)
vR2, vz2, vphi2 = model.intrinsic_moments(R=100, z=50)
major = model.los_second_moment(x=[10, 100, 300], y=0)
minor = model.los_second_moment(x=0, y=[10, 100, 300])
```

`jeanspy.axisymmetric` is a public module with immutable dataclass configurations;
use `dataclasses.replace` to change parameters. It deliberately does not inherit
the stateful spherical `Model` parameter/sampler contracts. This implements the
forward-model portion of [issue #52](https://github.com/gomeshun/jeanspy/issues/52).

Arrays broadcast; scalar inputs give scalar outputs. Negative or nonfinite
intrinsic second moments raise `ValueError`; they are not clipped. Valid density
parameters do not guarantee a physical Jeans solution or a nonnegative DF.
`beta_z < 1` alone is insufficient. The force API returns the **potential
gradient**, not acceleration. Exact cusp-origin force evaluation is unsupported;
the Jeans moments at the projected and intrinsic centers are supported.

The tracer density integrates to one with a `1/q` factor; its projected density
contains `1/q_projected`. These consistent normalizations cancel in the Jeans
ratios. For prolate tracers, x is still the line of nodes, not necessarily the
projected major axis. `intrinsic_axis_ratio(q_projected, inclination)` is an
oblate deprojection helper and rejects the face-on degeneracy. Direct intrinsic
q input supports face-on models.

The halo uses positive Zhao exponents `(alpha,beta,gamma)` with `alpha>0`,
`beta>2`, `0<=gamma<2`, and `Q>0`. The 2015 paper's halo is obtained with
`alpha=2, beta=3, gamma=-alpha_paper`; the standard NFW profile instead has
`alpha=1, beta=3, gamma=1`. No halo truncation or stellar self-gravity is added.

## Equations and numerics

In addition to the 2015 reference, see Hayashi et al. (2016),
https://arxiv.org/abs/1603.08046, equations (3)–(10) for the projection and Jeans
moments, (13)–(16) for the halo and force, and footnote 2 for quadrature.

Writing `P=nu*vz2`, the implemented equations are

```math
P(R,z)=\int_{|z|}^\infty \nu(R,z')\Phi_z(R,z')\,dz',\qquad
v_R^2=\frac{P}{(1-\beta_z)\nu},\qquad
\overline{v_\phi^2}=\frac{P+R P_R}{(1-\beta_z)\nu}+R\Phi_R.
```

The homoeoidal force integral uses `t=(1+tau/a0²)^(-1/2)`:

```math
D^2=1+(Q^2-1)t^2,\quad m^2=t^2(R^2+z^2/D^2),
\quad \Phi_R=4\pi GQR\int_0^1\frac{\rho(m)t^2}{D}\,dt,
\quad \Phi_z=4\pi GQz\int_0^1\frac{\rho(m)t^2}{D^3}\,dt.
```

The radial derivative of P is integrated analytically under the integral sign,
using the tracer derivative and halo logarithmic density slope. There is no
finite-difference step in the production solver. Infinite vertical and LOS
integrals use rational transformations; LOS quadrature is centered at the
maximum tracer density. The two LOS half-lines are integrated separately.
Defaults are 96 Gauss–Legendre nodes in each of the three integrals. Exposed
`n_force`, `n_vertical`, `n_los` allow independent refinement. These are fixed
orders, **not an adaptive accuracy guarantee**: compare increasing orders for
your scale ratios, steep cusps, strong flattening, and outer sky positions.
Runtime and memory grow approximately as their product per sky position.

## Validation and example

`python -m pytest tests/test_axisymmetric.py -q` checks:

- Exact isotropic self-consistent spherical Plummer intrinsic and LOS moments,
  from the center to ten scale radii, at three inclinations.
- Spherical NFW force and the existing spherical Jeans LOS solver.
- Analytic projected Plummer density against independent adaptive integration
  for oblate and prolate tracers and different inclinations.
- The Poisson equation and force parity for flattened/prolate NFW halos.
- Both Jeans equations using independent finite differences of pressure.
- An independent genuinely axisymmetric case: flattened Plummer tracer in a
  spherical Plummer potential viewed face-on. Swapping the Jeans/LOS integral
  order reduces it to the single adaptive reference integral
  `Sigma*vlos2 = 2*integral_0^infinity z*nu*Phi_z dz` with an analytic force.
- 64 versus 128 nodes for oblate and prolate halos, and sky-reflection symmetry.

These checks do not constitute reproduction of the paper's galaxy fits or a
JAM/Evans comparison; those remain useful additional external validations.

Run `python examples/axisymmetric_profiles.py` to print major/minor-axis
profiles and an explicit refinement check, or append `--output profiles.png`
to save a plot (requires matplotlib). It uses illustrative parameters only.
