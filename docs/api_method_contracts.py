"""Method-specific inputs, units and limits for the public API reference."""
from __future__ import annotations


def extend_methods(contracts):
    def method(classes, name, purpose, inputs, output, *, domain=None, errors=None, gradients=None):
        for cls in classes.split():
            base = contracts[cls].copy()
            base.update(Purpose=purpose, Inputs=inputs, Output=output)
            base["Errors"] = "Class-level validation notes (individual helpers may enforce only a subset): " + base["Errors"]
            if domain is not None:
                base["Domain"] = domain
            if errors is not None:
                base["Errors"] = errors
            if gradients is not None:
                base["Differentiation"] = gradients
            contracts[f"{cls}.{name}"] = base

    method("jeanspy.model.Model", "update", "Replace named parameters in the owning components.",
        "new_params is an optional mapping/Parameters/Series; keyword values are additional replacements. "
        "Names are physical names declared by this model and its components; target is ignored.",
        "None; mutates component parameters. params_all returns the resulting flattened copy.")
    method("jeanspy.model.Parameters", "to_series", "Convert parameter storage to a pandas Series.",
        "No arguments.", "Series named params, preserving key order; values keep their original units.")
    method("jeanspy.model.Parameters", "copy", "Make a shallow parameter copy.",
        "No arguments.", "New Parameters mapping; nested mutable values remain shared. Use copy.deepcopy for them.")

    classical_tracers = " ".join("jeanspy.model."+n for n in
        ["PlummerModel", "Exp2dModel", "Exp3dModel", "SersicModel", "Uniform2dModel"])
    for name, purpose, output in [
        ("density_2d", "Evaluate normalized projected tracer density.", "pc^-2 with input radius shape."),
        ("cdf_R", "Evaluate projected probability inside a circular radius.", "Dimensionless cumulative probability with input radius shape."),
    ]:
        method(classical_tracers, name, purpose, "R_pc is a scalar or NumPy array of projected radii in pc.", output)
    method(classical_tracers, "density_3d", "Evaluate the deprojected tracer density where implemented.",
        "r_pc is a scalar or NumPy array of intrinsic radii in pc. SersicModel also accepts method.",
        "pc^-3 with input radius shape; Uniform2dModel raises NotImplementedError.")
    implemented_tracers = classical_tracers.replace(" jeanspy.model.Uniform2dModel", "")
    method(implemented_tracers, "half_light_radius", "Return the projected half-light radius.",
        "No arguments; reads the stored tracer scales.", "Scalar radius in pc. Exp3dModel returns 1.67834699001666*re_pc.")
    method(implemented_tracers, "mean_density_2d", "Average surface density inside a circular aperture.",
        "Positive projected radius R_pc in pc, scalar or NumPy array.", "cdf_R(R)/(pi*R^2), in pc^-2 with radius shape.",
        domain="Use R>0; the elementary ratio is not a numerically regularized central limit.")
    method("jeanspy.model.PlummerModel jeanspy.model.Exp2dModel", "logdensity_2d",
        "Evaluate natural log projected density.", "R_pc in pc; scalar or NumPy array.",
        "Natural log of the numerical surface density in pc^-2, with input shape. "
        "This excludes the radial Jacobian 2*pi*R.")
    method("jeanspy.model.StellarModel", "density_2d_truncated", "Normalize a tracer within a finite aperture.",
        "R_pc is a nonnegative finite scalar/array in pc; R_trunc_pc is a positive finite scalar cutoff in pc.",
        "Normalized surface density in pc^-2 with radius shape, zero for R>R_trunc_pc.",
        domain="Requires a concrete density_2d and cdf_R with positive CDF at the cutoff.")
    for name, restriction in [
        ("density_3d_VM20", "0.5<=n<=10; 1e-3<=r/re<=1e3; positive finite r."),
        ("density_3d_VM20bis", "0.5<=n<=3.4; 1e-4<=r/re<=1e3; positive finite r."),
        ("density_3d_LGM", "Approximate deprojection; it is not a general reference or central-limit formula."),
        ("density_3d_auto", "Uses VM20bis, SP04 or numerical Abel inversion according to the documented n/r domain."),
    ]:
        method("jeanspy.model.SersicModel", name, "Evaluate the named Sersic deprojection.",
            "r_pc is scalar or an array in pc; n and re_pc come from stored parameters.",
            "Unit-integral tracer density in pc^-3 with input shape.", domain=restriction)
    method("jeanspy.model.SersicModel", "density_3d_numerical", "Compute numerical Abel deprojection.",
        "Nonnegative r_pc in pc; epsrel/epsabs are adaptive-integral tolerances; limit is the subdivision limit.",
        "pc^-3 with radius shape; infinity at r=0 for n>=1, a finite analytic center for n<1, zero at +infinity.",
        domain="Positive finite re_pc and supported interpolation-table n. At r>0 integrate over theta in [0,pi/2].",
        errors="Negative/NaN radii raise ValueError; SciPy convergence warnings can propagate. "
               "The returned value is not accompanied by a certified deprojection error.")

    halos = "jeanspy.model.NFWModel jeanspy.model.ZhaoModel"
    method(halos, "mass_density_3d", "Evaluate spherical halo density.",
        "r_pc in pc, scalar or NumPy array; reads the model's stored physical parameters.",
        "Msun/pc^3 with input shape. The density formula itself is untruncated; cusps can diverge at r=0.")
    for name in ("enclosed_mass", "enclosure_mass"):
        method(halos, name, "Evaluate halo mass inside a finite spherical radius.",
            "r_pc in pc, scalar or NumPy array; the Zhao implementation accepts n_steps.",
            "Msun within min(r_pc,r_t_pc), with input shape. enclosure_mass is the historical spelling.")
    for name, geometry in [
        ("jfactor_ullio2016", "Full finite-distance cone; 0<roi_deg<=90, dist_pc>r_t_pc."),
        ("jfactor_ullio2016_simple", "Small-aperture spherical approximation; outer shells projected into the cone are omitted."),
    ]:
        method("jeanspy.model.DMModel " + halos, name, "Evaluate an annihilation factor with explicit geometry.",
            "dist_pc is observer distance in pc; roi_deg is cone half-angle in degrees. "
            "Scalar inputs are the usual case; mutually broadcastable arrays are supported by these classical helpers.",
            "J in GeV^2 cm^-5, with broadcast geometry shape.", domain=geometry+
            " Require a positive finite halo cutoff and a convergent inner cusp. "
            "Small-angle variants enforce the configured roi_deg_max_warning bound.")
    method("jeanspy.model.NFWModel", "jfactor_evans2016", "Evaluate the historical infinite-LOS NFW approximation.",
        "dist_pc in pc and roi_deg in degrees, positive finite broadcastable values; stored NFW scales/cutoff.",
        "J in GeV^2 cm^-5, scalar for scalar geometry.",
        domain="Small-angle formula: the projected aperture is capped at r_t_pc but the LOS density is untruncated. "
               "It is a different integral from the three-dimensionally truncated finite-cone factor.")

    anisotropies = " ".join("jeanspy.model."+n for n in
        ["ConstantAnisotropyModel", "OsipkovMerrittModel", "BaesAnisotropyModel"])
    for name, purpose, output in [
        ("beta", "Evaluate spherical velocity anisotropy.", "Dimensionless beta; constant models may return a scalar."),
        ("f", "Evaluate the radial Jeans integrating factor.", "An arbitrarily normalized integrating factor with radius shape."),
    ]:
        method(anisotropies, name, purpose, "r is a radius or NumPy radius array in pc.", output)
    method(anisotropies, "kernel", "Evaluate the spherical LOS projection kernel.",
        "u=r/R>=1 is dimensionless; R is projected radius in pc; broadcastable arrays. "
        "The Baes numerical implementation takes fixed quadrature n through kwargs.",
        "Dimensionless projection kernel with the broadcast shape.")
    for name, component in [("sigmar2", "radial"), ("sigmat2", "one-component tangential")]:
        method("jeanspy.model.DSphModel", name, f"Compute intrinsic {component} velocity variance.",
            "r_pc is a scalar or NumPy array of positive intrinsic radii in pc.",
            "Variance in (km/s)^2, following input shape; scalar input is a zero-dimensional NumPy array.",
            domain="Uses adaptive integration to infinite radius with vanishing outer pressure. "
                   "The intrinsic helpers do not apply all LOS input validation checks.")
    for name in ["sigmalos2", "sigmalos2_dequad", "sigmalos_dequad"]:
        method("jeanspy.model.DSphModel", name, "Project a spherical velocity moment along the line of sight.",
            "R_pc is positive finite scalar or nonempty one-dimensional pc array; n is the outer fixed-rule order "
            "and n_kernel the anisotropy-kernel order.",
            "Variance in (km/s)^2, or dispersion in km/s for sigmalos_dequad; scalar for scalar input, otherwise (N,).")

    for cls, coords in [("jeanspy.model.AxisymmetricDSphModel", "x_pc/y_pc"),
                        ("jeanspy.model_numpyro.AxisymmetricDSphModel", "x_pc/y_pc")]:
        method(cls, "sigmalos2", "Project a cylindrically aligned second moment.",
            f"Signed {coords} in pc, broadcastable scalar/arrays; params is the explicit physical dictionary.",
            "LOS second moment in (km/s)^2 with the broadcast coordinate shape, including scalar output.")
        method(cls, "intrinsic_moments", "Evaluate the intrinsic Jeans second moments.",
            "R_pc>=0 and signed z_pc in pc, broadcastable; params supplies the physical dictionary.",
            "Tuple (vR2,vz2,vphi2), each in (km/s)^2 with the broadcast coordinate shape. "
            "vphi2 is the total azimuthal second moment; no rotation/dispersion split is assigned.")
        method(cls, "potential_gradient", "Evaluate derivatives of the gravitational potential.",
            "R_pc>=0, signed z_pc in pc and explicit params.",
            "Tuple (dPhi/dR,dPhi/dz) in (km/s)^2/pc; gravitational acceleration has the opposite sign.")
        method(cls, "surface_density", "Evaluate the projected spheroidal Plummer tracer.",
            "Signed x_pc/y_pc in pc and explicit params; coordinates broadcast.",
            "Normalized surface density in pc^-2 with the broadcast coordinate shape.")
        method(cls, "enclosed_mass", "Integrate mass inside a similar halo ellipsoid.",
            "m_pc>=0 is the ellipsoidal radius in pc; params supplies halo scales, slopes, Q and cutoff.",
            "Msun inside R^2+z^2/Q^2<=min(m_pc,r_t_pc)^2, matching m_pc shape.")
    for name, units in [("jfactor", "GeV^2 cm^-5"), ("dfactor", "GeV cm^-2")]:
        method("jeanspy.model.AxisymmetricDSphModel", name, "Postprocess an axisymmetric finite-cone factor.",
            "Scalar dist_pc and roi_deg; params specifies halo, inclination and finite r_t_pc; "
            "n_mu/n_phi/n_radial set independent factor quadratures, all >=16.",
            "Python float in "+units+".", domain=contracts["jeanspy.axisymmetric_factors."+name]["Domain"])

    for cls in ["jeanspy.model_numpyro.NFWModel", "jeanspy.model_numpyro.ZhaoModel"]:
        method(cls, "mass_density_3d", "Evaluate functional spherical halo density.",
            "r_pc is scalar/array in pc; params is a physical scalar dictionary.",
            "Untruncated density in Msun/pc^3 with radius shape.")
        for name in ["enclosed_mass", "enclosure_mass", "enclosed_mass_analytic", "enclosed_mass_numeric"]:
            method(cls, name, "Evaluate functional spherical halo mass.",
                "r_pc is scalar/array in pc; params is the physical dictionary. "
                "enclosed_mass/enclosure_mass select method=auto/analytic/numeric; numerical methods accept n_steps.",
                "Mass in Msun within min(r_pc,r_t_pc), matching radius shape. Invalid dynamic proposals yield NaN.")
    for name in ["sigmalos2", "sigmalos2_kernel", "sigmalos2_abel"]:
        method("jeanspy.model_numpyro.DSphModel", name, "Evaluate functional spherical LOS variance.",
            "Positive R_pc in pc, scalar or nonempty 1-D array; params contains physical scalars. "
            "Use the signature's static numerical/backend options; n_u and n_kernel apply to the kernel route, "
            "n_r and u_max to the Abel grid, and dm_mass_n_steps to the mass integral.",
            "Always a one-dimensional array of variances in (km/s)^2, length one for scalar input.")
    method("jeanspy.model_numpyro.PlummerModel", "sample_R", "Sample projected Plummer radii.",
        "key is a JAX random key; n is a static nonnegative count; re_pc is positive scale in pc.",
        "Array (n,) of radii in pc. Split keys explicitly before repeated independent draws.")
    method("jeanspy.model_numpyro.PlummerModel", "log_prob_R", "Evaluate normalized projected radial log probability.",
        "R_pc and positive re_pc in pc, broadcastable.",
        "Natural log radial PDF including 2*pi*R, with broadcast shape; minus infinity off support.")

    inference = "jeanspy.model.SimpleDSphEstimationModel jeanspy.model.AxisymmetricDSphEstimationModel"
    for name, output in [("lnlikelihoods", "Per-star log densities, shape (N,)."),
                          ("lnlikelihood", "Scalar sum of log likelihoods."),
                          ("lnpriors", "Sequence of log prior contributions in prior_names order."),
                          ("lnposterior", "Tuple (logposterior, loglikelihood, *prior_terms) for emcee blobs."),
                          ("lnposterior_wbic", "Tuple using loglikelihood/log(N) plus the original prior; requires N>1.")]:
        method(inference, name, "Evaluate the explicit kinematic inference target.",
            "p is one parameter vector of shape (ndim,) in p_names_lnprob order. Uses already loaded observations.", output)
    method(inference, "convert_params", "Map sampling coordinates to physical parameters.",
        "One parameter vector in exact prior order; log10_ and bfunc_ prefixes identify the supported transforms.",
        "Named physical parameters with pc, Msun/pc^3, km/s, radians and dimensionless quantities as appropriate. "
        "The axisymmetric result also incorporates fixed_params.")
    method("jeanspy.model.FlatPriorModel", "sample", "Draw from the finite sampling-coordinate bounds.",
        "size is a sample count, tuple of sample axes or None; uses NumPy's global random state.",
        "Uniform coordinates with trailing parameter axis; size=None returns one vector.")
    method("jeanspy.model.PhotometryPriorModel", "sample", "Draw a log-radius prior value.",
        "size is a sample count/shape or None, following SciPy normal-distribution sampling.",
        "Samples in log10(pc), not physical radii; shape follows size.")

    sampler = "jeanspy.sampler_numpyro.NumPyroSampler"
    for name, purpose, inputs, output in [
        ("run", "Run one chunk and optionally persist it.", "rng_key plus model data arguments; resume is auto/True/False. "
         "save_checkpoint/save_samples/wait_for_write control persistence; arviz_kwargs are conversion options.",
         "SamplerRunResult; wait for flush/close before relying on asynchronous files."),
        ("save_checkpoint", "Save trusted NumPyro transition state.", "No arguments; requires a last_state and bound analysis identity.", "Path of the written checkpoint."),
        ("load_checkpoint", "Load a trusted matching transition state.", "No arguments; reads this output directory's checkpoint and metadata.", "Checkpoint Path; sets post_warmup_state only after identity checks."),
        ("save_samples_chunk", "Write an ArviZ sample chunk.", "Optional datatree; wait=True writes synchronously; other keywords go to the converter.", "Tuple (chunk_index, path, write_submitted). False in the third entry means the write was synchronous."),
        ("load_samples", "Read persisted sample chunks.", "combine=True concatenates matching draw dimensions; False retains separate trees.", "Combined xarray.DataTree or ordered list of DataTrees. Pending writes are flushed first."),
        ("combine_trees", "Combine compatible ArviZ chunks.", "Nonempty sequence of DataTrees from the same analysis, with consistent groups, variables and nondraw coordinates.", "DataTree concatenated over draws, with draw coordinates renumbered from zero. No independent sampling-identity verification occurs here."),
        ("flush", "Wait for outstanding sample writes.", "No arguments.", "None; asynchronous exceptions propagate."),
        ("close", "Finish writes and shut down the background writer.", "No arguments; also called by context-manager exit.", "None; failures propagate."),
        ("clear_resume_state", "Clear the in-memory warmup-resume pointer.", "No arguments.", "None; sets mcmc.post_warmup_state=None. It does not delete or reset persisted analysis files."),
        ("to_datatree", "Convert the wrapped MCMC result to ArviZ storage.", "Keyword arguments forwarded to arviz_converter.", "xarray.DataTree with chain/draw dimensions; the converter must return this format."),
        ("list_chunk_paths", "Find sample chunks in index order.", "No arguments.", "List of Paths; an empty list means no saved chunks were found."),
    ]:
        method(sampler, name, purpose, inputs, output)
    domains = {
        "run": "Use an unchanged model, prior, observations and numerical configuration when resuming. Identity is checked before sampling.",
        "save_checkpoint": "A completed MCMC last_state and an unchanged bound analysis are required. The file contains Python pickle data.",
        "load_checkpoint": "Read only a trusted checkpoint. Stored target/metadata identity is checked here; supplied observation arguments are checked later by run.",
        "save_samples_chunk": "The sampler assigns a fresh chunk index. Wait for flush/close before relying on an asynchronous write.",
        "load_samples": "Reads this directory's chunks. Completed arrays are loaded into memory before closing backing stores.",
        "combine_trees": "Supply chunks from the same analysis. Static groups must be equal. The draw-group concatenation uses xarray compat=override and is not a general identity validator.",
        "flush": "Waits for outstanding writes and propagates their failures; it does not change the MCMC target or validate a checkpoint.",
        "close": "Finish pending writes before releasing the executor. Prefer a context manager to ensure this happens.",
        "clear_resume_state": "Affects only the in-memory post_warmup_state pointer. A later auto-resume can still use last_state or an on-disk checkpoint.",
        "to_datatree": "The wrapped MCMC must contain a result supported by the chosen converter.",
        "list_chunk_paths": "Lists recognized chunk names in this directory; it does not inspect their scientific content or verify identity.",
    }
    errors = {
        "run": "ValueError for identity mismatch or invalid resume mode; FileNotFoundError for required missing state. Model, NumPyro and storage exceptions propagate.",
        "save_checkpoint": "RuntimeError before last_state exists; ValueError for unverified/changed analysis. Filesystem errors propagate.",
        "load_checkpoint": "FileNotFoundError for a missing checkpoint; ValueError for format/identity mismatch. Pickle and filesystem errors propagate.",
        "save_samples_chunk": "FileExistsError prevents replacing a reserved chunk. Conversion, array loading and write failures propagate; asynchronous failures are raised by flush/close.",
        "load_samples": "FileNotFoundError when no chunks exist. Pending-write, reader and concatenation errors propagate.",
        "combine_trees": "ValueError for an empty list, missing groups, inconsistent draw-axis presence or changed static groups. Other xarray errors propagate.",
        "flush": "A pending write's exception is re-raised after waiting for the submitted futures.",
        "close": "Write failures propagate after the executor is shut down.",
        "clear_resume_state": "No additional validation or documented domain exception.",
        "to_datatree": "TypeError if the converter returns anything other than xarray.DataTree; converter exceptions propagate.",
        "list_chunk_paths": "Filesystem access errors may propagate; an empty directory returns an empty list.",
    }
    for name in domains:
        contracts[f"{sampler}.{name}"].update(Domain=domains[name], Errors=errors[name],
            Differentiation="Host-side sampling/storage control; this method has no physical-parameter derivative.")
