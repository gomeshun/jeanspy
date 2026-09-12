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
