# Classical backend structure

The public stateful NumPy/SciPy API remains `jeanspy.model`. Its implementation is split into focused private modules under `jeanspy._classical`:

- `core.py`: `Parameters` and the base `Model` composition/parameter machinery.
- `profiles.py`: stellar-density, dark-matter, and anisotropy profile components.
- `jfactor.py`: J-factor units and Ullio & Valli geometry helpers.
- `solver.py`: the composite `DSphModel` and classical Jeans-equation solvers.
- `inference.py`: priors, data handling, likelihoods, and classical estimation helpers.

`jeanspy.sersic` remains separate because its deprojection implementation and coefficient tables have their own maintenance/validation lifecycle.

`jeanspy.model` explicitly re-exports the supported public classes and functions. `_model_impl.py` is now only a compatibility shim for code that imported that historical private module directly; new code should not depend on it.

The intended dependency direction is:

```text
core
  ├── jfactor
  └── profiles
        └── solver
              └── inference

sersic ──> profiles.StellarModel
model  ──> explicit public exports
```

This structure is deliberately private below `jeanspy.model`: downstream code should continue to import supported classical APIs from `jeanspy.model`, not from `jeanspy._classical`.
