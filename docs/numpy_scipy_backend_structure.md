# NumPy/SciPy backend structure

The public stateful NumPy/SciPy API remains `jeanspy.model`. Its implementation is split into focused private modules under `jeanspy._numpy`:

- `core.py`: `Parameters` and the base `Model` composition/parameter machinery.
- `profiles.py`: stellar-density, dark-matter, and anisotropy profile components.
- `jfactor.py`: J-factor units and Ullio & Valli geometry helpers.
- `solver.py`: the composite `DSphModel` and NumPy/SciPy Jeans-equation solvers.
- `inference.py`: priors, data handling, likelihoods, and NumPy/SciPy estimation helpers.

`jeanspy.sersic` remains separate because its deprojection implementation and coefficient tables have their own maintenance/validation lifecycle.

`jeanspy.model` explicitly re-exports the supported public classes and functions.
The temporary `_model_impl` forwarding module has been removed. There are no
private compatibility import routes; use `jeanspy.model` for supported imports.

The direct internal imports use these dependencies (`A -> B` means A imports B):

```text
profiles -> core, jfactor
solver -> core, profiles
inference -> core, profiles, solver, parameters
sersic -> profiles.StellarModel
axisymmetric_inference -> inference priors
axisymmetric_factors -> jfactor units
model -> explicit public exports
```

This structure is deliberately private below `jeanspy.model`: downstream code should continue to import supported NumPy/SciPy APIs from `jeanspy.model`, not from `jeanspy._numpy`.

The spherical inference interface is `SphericalDSphEstimationModel`.
`plummer_nfw_constant_anisotropy_model` supplies the named profile composition
and requires explicit prior bounds. Resume identity format 2 compares computational
code and target state; full source bytes are retained as separate provenance.
See [the API migration guide](source/guides/api-migration.md).
