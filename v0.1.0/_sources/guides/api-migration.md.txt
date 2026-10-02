# Explicit model names and parameter specifications

The following API changes replace ambiguous names with their computational or
physical meaning. Update imports and parameter dictionaries together with the
examples below. The old spellings are no longer supported.

| Previous API | Current API |
| --- | --- |
| `jeanspy.model_numpyro` | `jeanspy.model_jax` for spherical JAX forward calculations |
| `jeanspy.axisymmetric_numpyro` | `jeanspy.axisymmetric_jax` for axisymmetric JAX forward calculations |
| `Exp2dModel(re_pc=re)` | `ProjectedExponentialModel(r_exp_pc=re / 1.67834699001666)` |
| `Exp3dModel(re_pc=scale)` | `ProjectedExponentialModel(r_exp_pc=scale)` |
| Spherical Zhao `a`, `b`, `g` | `alpha`, `beta`, `gamma`, matching axisymmetric Zhao |
| `Model.update(..., target=...)` | Remove `target`; only declared physical parameter names are accepted |
| `jfactor_ullio2016` | `jfactor_cone` |
| `jfactor_ullio2016_simple` | `jfactor_spherical_aperture` |
| `NFWModel.jfactor_evans2016` | `jfactor_small_angle_infinite_los` |
| `roi_deg_max_warning` | `small_angle_limit_deg`, an enforced limit |
| `assert_roi_is_enough_small` | `validate_small_angle` |
| JAX `sigmalos2(backend=...)` | `sigmalos2(solver=...)`, selecting `auto`, `kernel` or `abel` |
| JAX `sigmalos2(constant_kernel_backend=...)` | `sigmalos2(kernel_backend=...)`, selecting `jax` or `scipy` for constant anisotropy |
| Constant-anisotropy `kernel(backend=...)` | `kernel(kernel_backend=...)` |
| Axisymmetric example `--backend numpy/numpyro` | `--sampler emcee/numpyro` |
| `get_default_estimation_model` | `plummer_nfw_constant_anisotropy_model`, with required `config` |
| `SimpleDSphEstimationModel` | `SphericalDSphEstimationModel` for custom spherical compositions |
| `FlatPriorModel.generate_default_config_file` | `FlatPriorModel.write_config_template` |
| Sérsic `method="approx"` / `deprojection_method="approx"` | `"lgm"` (Lima Neto--Gerbal--Márquez) |
| Sérsic `norm_3d` / `b_approx` | `lgm_norm_3d` / `b_linear` |
| `enclosure_mass` | `enclosed_mass` in both backends |
| `inverse_temparature` | `inverse_temperature` in both inference interfaces |
| NumPy `sigmalos2_dequad` / `sigmalos_dequad` | `sigmalos2(..., method="dequad")` / `sigmalos(..., method="dequad")` |

`jeanspy.sampler_numpyro` remains the NumPyro inference and storage module.
`get_runtime_config()` now reports `sigmalos2_solver_default` and
`kernel_backend_default`; the exported solver constant is
`DEFAULT_SIGMALOS2_SOLVER`. `storage_backend` still selects a storage format.

`ProjectedExponentialModel.re_pc` is a read-only property giving the projected
half-light radius. Its stored physical parameter is `r_exp_pc`, the scale
inside the projected exponential. Update that parameter to change the model.

## Specify sampling coordinates

NumPy/SciPy estimation models accept `parameter_specs` in prior-table order.
Each `SamplingParameter` specifies a sampled name, physical name and transform.
Without specifications, names map by identity; prefixes never trigger a transform.

```python
from jeanspy.parameters import SamplingParameter

parameter_specs = [
    SamplingParameter("vmem_kms", "vmem_kms"),
    SamplingParameter("log10_re_pc", "re_pc", "pow10"),
    SamplingParameter("log10_rs_pc", "rs_pc", "pow10"),
    SamplingParameter("log10_rhos_Msunpc3", "rhos_Msunpc3", "pow10"),
    SamplingParameter("log10_r_t_pc", "r_t_pc", "pow10"),
    SamplingParameter("log10_one_minus_beta_ani", "beta_ani", "one_minus_pow10"),
]
# Pass these to SphericalDSphEstimationModel(parameter_specs=..., ...).
# Prior-table row labels must match the sample_name values, in this order.
```

The preset `plummer_nfw_constant_anisotropy_model` supplies exactly these specifications.
Rename its prior-table row `bfunc_beta_ani` to `log10_one_minus_beta_ani` without
changing the numerical bounds. Those bounds are uniform in `log10(1-beta_ani)`.
For custom compositions, coordinate labels can be arbitrary; the specifications
determine the meaning. `one_minus_pow10` returns `1-10**x`, not `10**x`.

Axisymmetric models use the same `SamplingParameter` class. An explicit
`SamplingParameter("cos_inclination", "inclination", "arccos")` maps a cosine
coordinate to radians. A photometric prior requires a `pow10` coordinate for
`re_pc`. For a spherical projected exponential, specify `pow10` for `r_exp_pc`;
the estimation model converts this scale to `re_pc` before evaluating the
photometric prior and when drawing initial samples.

If an `Exp2dModel` prior was uniform in `log10(re_pc)`, subtract
`log10(1.67834699001666)` from its bounds when moving to a `log10(r_exp_pc)`
coordinate. Keep the photometric prior's location and width in `log10(re_pc)`.
Old `Exp3dModel` fits used a parameter called `re_pc` that actually represented
the exponential scale; check the interpretation of any photometric prior used
with those fits. The new model consistently applies that prior to half-light radius.

NumPyro retains its explicit, distribution-bearing `ParameterSpec`. Priors in
both interfaces remain densities in sampled coordinates; conversion adds no
implicit Jacobian and does not change a log-uniform prior into a linear-uniform one.

## Density cutoffs and observation precision

Spherical NFW and Zhao density now vanish for `r > r_t_pc`, matching their
enclosed mass and finite-cone factors. The cutoff boundary is included, and
`r_t_pc=np.inf` retains an untruncated halo at finite radii. Custom integrations
using `mass_density_3d` therefore change outside a finite cutoff. The three
[J-factor methods](factors.md) retain their distinct integration geometries.

Spherical observations no longer unconditionally convert to float32.
`dtype=None` preserves the common input floating dtype; integer-only inputs use
float64. An explicit `dtype=` selects storage precision, including shared memory.
Shared data cannot change shape or dtype on reset; construct a new model for
that change. Dtype and parameter specifications participate in sampling identity.

These changes alter the identity of a sampling target. Existing chains remain
readable with their original metadata, but the modified package must use a new
output directory rather than resume a chain created with the previous source.
Retain the original checkout and environment to reproduce or resume that analysis.


## Explicit presets and public utilities

`plummer_nfw_constant_anisotropy_model(data, photometry_prior_loc,
photometry_prior_scale, config)` composes Plummer light, NFW mass, constant
anisotropy, Gaussian LOS velocities, uniform coordinate bounds and a Gaussian
photometric prior in `log10(re_pc)`. A missing CSV raises `FileNotFoundError`
without writing a file. Use `FlatPriorModel.write_config_template` explicitly
if a blank prior table is wanted; complete its bounds before constructing a model.

Private implementation imports are unsupported. The temporary `_model_impl`
module has been removed and the NumPy/SciPy implementation is organized under
`_numpy`; import supported classes from `jeanspy.model`.
`dequad` and `generate_x_w` remain public. Memoization/hashability helpers and
hypergeometric quadrature containers are private. API documentation now requires
an explicit `__all__` in every public module, preventing incidental helper exports.

## Resume compatibility and source provenance

The identity format is now **2**. A new output location is required for every
format-1 chain and for the API changes above. Preserve the original code and
environment to continue those analyses; no old identity is silently replaced.

Within format 2, edits to comments, code layout and docstrings do not by themselves
invalidate a chain. The package comparison uses Python syntax with only those
documentary elements removed. Runtime callable code, defaults and captured state,
model parameters, observations, priors, coordinate order/transforms, solver and
sampler settings, packaged data, Python/dependency versions, and JAX backend and
precision are still checked. Adding/removing/renaming package modules also changes
identity. Checks are recomputed at persistence boundaries, including repeated
runs in the same process.

Full source/data byte hashes remain recorded separately, with a new history entry
when an accepted run uses changed bytes. NumPyro writes `source_provenance` in
`metadata.json`; emcee stores JSON records in the `jeanspy_source_provenance`
dataset of its HDF5 backend group, with the starting iteration for each record.
These hashes document which files were present; compatibility is determined by
the analysis identity, not by substituting provenance records.

This guards accidental mismatches, not arbitrary Python side effects. Custom
models must expose external/opaque state through `sampling_identity()`. If a
model uses documentation or source text as computational input, include that text
in its declared identity too. Preserve the matching files/environment for exact
reproduction; a hash alone is not a source archive.
