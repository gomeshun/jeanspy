"""Additional API contracts rendered by Sphinx without changing runtime objects.

Keep the physical statements here aligned with the implementation and feature
guides. These notes supplement, rather than replace, the real signatures and
method docstrings. Values use plain scalars/arrays, not Astropy Quantity objects.
"""
from __future__ import annotations

CONTRACTS = {}


def add(paths, purpose, inputs, output, domain, errors, backend, gradients, example):
    for path in paths.split():
        CONTRACTS[path] = dict(Purpose=purpose, Inputs=inputs, Output=output,
                              Domain=domain, Errors=errors, Backend=backend,
                              Differentiation=gradients, Example=example)


CLASSICAL = "NumPy/SciPy CPU; stateful components, with no JAX tracing."
JAX = "JAX arrays on the configured CPU/GPU, with dtype set before import."
NO_GRAD = "No physical-parameter automatic differentiation on this API."
PROFILE_DOMAIN = ("Supply positive finite scales and real nonnegative radii. "
                  "Use NumPy arrays for array inputs. A central cusp may diverge at zero. "
                  "The older elementary density formulas do not uniformly validate domains.")
PROFILE_ERRORS = ("Unknown parameter names raise ValueError in Model.update. "
                  "Invalid values in elementary profile formulas can produce NaN/inf; "
                  "successful construction alone does not validate a physical profile.")
SCALAR_PARAMS = ("Physical parameter dictionaries hold scalar values; radius arrays "
                 "are broadcast independently. Use vmap to batch parameter dictionaries.")

add("jeanspy.model.Model", "Compose and update classical model components.",
    "show_init is a logging flag; submodels maps required role names to component instances; "
    "**params sets scalar physical parameters. Subclasses declare required names and roles.",
    "A mutable model; model[role] returns its submodel. update(new_params, **kwargs) mutates "
    "the owning components and returns None. params_all is a flattened Parameters copy; "
    "params_all_with_model_name retains role-qualified names.",
    "This is a subclassing interface. Unspecified parameters are initialized to NaN; "
    "supply all physical values before numerical evaluation. target is retained but ignored by update.",
    "Missing subclass declarations raise AttributeError; mismatched roles or unknown parameters "
    "raise ValueError.", CLASSICAL, NO_GRAD, "See the spherical quickstart and composition guide.")
add("jeanspy.model.Parameters jeanspy.model.DotDict",
    "Store named state with mapping and historical attribute access.",
    "An optional mapping plus keyword values; keys are strings and units belong to the stored values.",
    "Parameters.copy is shallow; deepcopy separates nested values. Parameters.index and .values "
    "are lists, not NumPy arrays or dict methods; to_series returns a pandas Series. "
    "DotDict follows dict operations; assignment to an existing key through an attribute changes the key.",
    "Container operations have no physical validation. DotDict new attributes need not become keys.",
    "Missing mapping keys raise KeyError; missing attributes raise AttributeError.",
    "Python host-side containers.", NO_GRAD, "Parameters({'re_pc': 300.}).to_series().")

for name, detail, params in [
    ("PlummerModel", "Unit-integral Plummer tracer; re_pc is its projected half-light radius.",
     "re_pc (pc)."),
    ("Exp2dModel", "Unit-integral projected exponential tracer with an Abel-deprojected K0 density.",
     "re_pc is the projected half-light radius (pc); R_exp_pc = re_pc/1.67834699001666."),
    ("Exp3dModel", "Historical projected exponential tracer with a different scale convention.",
     "re_pc is the exponential scale length (pc), despite its name; "
     "half_light_radius() is 1.67834699001666*re_pc. It is not a pure 3-D exponential."),
    ("Uniform2dModel", "Unit-integral uniform projected disk with no three-dimensional tracer.",
     "Rmax_pc is the disk radius (pc)."),
    ("SersicModel", "Unit-integral Sersic tracer with explicit deprojection choices.",
     "re_pc is projected half-light radius (pc); n is dimensionless Sersic index; "
     "deprojection_method is auto/approx/vm20/vm20bis/numerical."),
]:
    add("jeanspy.model."+name, detail, params + " Other constructor arguments follow Model.",
        "density_2d and density_3d return pc^-2 and pc^-3 with input shape. "
        "cdf_R returns the dimensionless projected radial CDF; half_light_radius returns pc. "
        "mean_density_2d is mean surface density within R. "
        "logdensity_2d is the natural log of the surface density, not the radial PDF. "
        "density_2d_normalized_re, where available, is the dimensionless ratio Sigma(R)/Sigma(re).",
        PROFILE_DOMAIN + (" Uniform2dModel returns zero outside the disk; it cannot feed a 3-D Jeans solver."
                          if name == "Uniform2dModel" else ""),
        PROFILE_ERRORS + (" density_3d raises NotImplementedError."
                          if name == "Uniform2dModel" else ""),
        CLASSICAL, NO_GRAD, "examples/docs_profiles.py")
add("jeanspy.model.StellarModel", "Subclassing interface for normalized stellar profiles.",
    "Subclasses define density_2d(R_pc), density_3d(r_pc) and required parameters. "
    "density(distance_from_center, dimension) dispatches on '2d'/'3d'. "
    "density_2d_truncated(R_pc, R_trunc_pc) requires a scalar positive cutoff in pc.",
    "Density in pc^-2/pc^-3; truncated surface density integrates to one within the cutoff "
    "and is zero outside. Shape follows radius input.",
    "A 3-D density is required by a Jeans solver. The truncated surface-density helper "
    "also requires cdf_R in the concrete profile.",
    "Abstract methods raise NotImplementedError; invalid dimension, radii or truncation raise ValueError.",
    CLASSICAL, NO_GRAD, "examples/docs_profiles.py")

for name, params in [
    ("NFWModel", "rs_pc (pc), rhos_Msunpc3 (Msun/pc^3), r_t_pc (pc)."),
    ("ZhaoModel", "rs_pc (pc), rhos_Msunpc3 (Msun/pc^3), a/b/g (dimensionless transition, "
     "outer and inner slopes), r_t_pc (pc)."),
]:
    add("jeanspy.model."+name, "Spherical dark-matter density and finite-radius mass.",
        params + " r_pc is scalar or an array in pc; n_steps controls numerical Zhao mass integration.",
        "mass_density_3d returns Msun/pc^3 with input shape; enclosed_mass returns Msun "
        "inside min(r_pc, r_t_pc). enclosure_mass is a historical alias. "
        "The classical density method itself evaluates the untruncated profile.",
        "Positive scales and cutoff; Zhao a>0 and g<3 for finite central mass; finite-radius mass "
        "does not require b>3. Total untruncated mass can diverge. J-factor requires g<1.5.",
        ("Invalid Zhao mass domains raise ValueError. " if name == "ZhaoModel" else
         "NFW's elementary mass formula does not uniformly validate physical domains. ") +
        "Elementary density calculations may return NaN/inf. "
        "J-factor methods validate geometry and quadrature separately.",
        CLASSICAL, NO_GRAD, "examples/docs_profiles.py; examples/docs_factors.py")
add("jeanspy.model.DMModel", "Subclassing interface for spherical dark matter and J factors.",
    "Subclasses supply mass_density_3d and enclosed_mass/enclosure_mass. "
    "J-factor methods take scalar dist_pc and roi_deg (cone half-angle in degrees).",
    "Mass in Msun; density in Msun/pc^3; J factor in GeV^2 cm^-5.",
    "J-factor geometry requires a finite positive r_t_pc and an external observer for the full cone. "
    "The simple method omits shells outside the spherical aperture; see the factors guide.",
    "Invalid scales/apertures, divergent cusps or failed adaptive quadrature raise ValueError.",
    CLASSICAL, NO_GRAD, "examples/docs_factors.py")

for name, params in [
    ("ConstantAnisotropyModel", "beta_ani is dimensionless and constant."),
    ("OsipkovMerrittModel", "r_a is a positive anisotropy radius in pc; beta(r)=r^2/(r^2+r_a^2)."),
    ("BaesAnisotropyModel", "beta_0 and beta_inf are inner/outer anisotropies; r_a (pc) is positive; eta>0 sets transition sharpness."),
]:
    add("jeanspy.model."+name, "Spherical velocity-anisotropy profile and projection kernel.",
        params+" beta(r_pc) and f(r_pc) take radii in pc. kernel(u, R_pc, n) uses u=r/R>=1 "
        "and projected radius R_pc in pc; n is a fixed quadrature order.",
        "beta is dimensionless; f is an arbitrarily normalized integrating factor satisfying "
        "d ln(f)/d ln(r)=2 beta. The dimensionless kernel broadcasts u and R inputs.",
        "Require beta<1 for a positive tangential dispersion; this is necessary, not sufficient "
        "for a nonnegative global distribution function. Check numerical convergence near limits.",
        "Elementary formulas do not uniformly validate all physical domains; invalid values "
        "can produce nonfinite results later rejected by the LOS solver.",
        CLASSICAL, NO_GRAD, "examples/docs_profiles.py")
add("jeanspy.model.AnisotropyModel", "Subclassing interface for spherical anisotropy.",
    "Implement beta(r_pc), f(r_pc) and kernel(u,R_pc,n) consistently; radii use pc and u=r/R.",
    "Dimensionless beta/kernel and an arbitrary-normalization integrating factor f.",
    "Steady spherical Jeans closure; beta is distinct from axisymmetric beta_z.",
    "Abstract methods raise NotImplementedError.", CLASSICAL, NO_GRAD, "examples/docs_profiles.py")
add("jeanspy.model.DSphModel", "Combine a spherical tracer, dark halo and anisotropy into velocity moments.",
    "submodels must contain StellarModel, DMModel and AnisotropyModel; vmem_kms sets mean velocity. "
    "sigmalos2_dequad uses n outer and n_kernel inner nodes; sigmalos2 dispatches to that same route. "
    "R_pc is scalar or nonempty 1-D projected radius; r_pc is intrinsic radius (pc).",
    "sigmar2 and sigmat2 return intrinsic radial and one-component tangential variances in (km/s)^2. "
    "sigmalos2 and sigmalos2_dequad return LOS variance arrays with input shape, or a scalar for scalar input. "
    "sigmalos_dequad returns km/s. integrand_sigmalos2(u,R_pc) has shape (N_R,N_u).",
    "Finite positive projected radii; no central-limit LOS solver. Tracer has vanishing outer pressure. "
    "Numerical orders and adaptive tolerances are part of the analysis configuration.",
    "Malformed/invalid projected radii or nonphysical mass, density and integrand values raise ValueError. "
    "Adaptive reference integration may issue SciPy integration warnings.",
    CLASSICAL, NO_GRAD, "examples/docs_spherical.py")

add("jeanspy.model.FlatPriorModel", "Finite uniform bounds in named sampling coordinates.",
    "config is a pandas DataFrame indexed by ordered parameter names, with finite lower/upper columns, "
    "or a CSV path. sample(size) uses NumPy's random state. generate_default_config_file writes a CSV template.",
    "A validated prior object; sample returns coordinates with trailing parameter axis. "
    "lower/upper are array copies. extract_value_by_name expects exactly one parameter vector.",
    "Unique nonempty names and lower<upper. Bounds apply before log/power transforms. "
    "Unfilled default NaN bounds are intentionally unusable for inference.",
    "Invalid schema/bounds/vector shape raise ValueError or TypeError; missing CSV raises FileNotFoundError.",
    CLASSICAL, NO_GRAD, "examples/docs_inference.py")
add("jeanspy.model.PhotometryPriorModel", "Gaussian prior in log10(re_pc).",
    "loc and scale are location and standard deviation in log10(pc); sample(size) uses SciPy's random state. "
    "reset_prior(loc,scale) replaces that distribution.",
    "A prior object; sample returns log10 radii, not physical pc.",
    "Finite loc and positive finite scale are required by the estimation model.",
    "Invalid prior values are rejected when composing SimpleDSphEstimationModel.",
    CLASSICAL, NO_GRAD, "examples/docs_inference.py")
add("jeanspy.model.FittableModel jeanspy.model.SimpleDSphEstimationModel",
    "Classical kinematic inference with explicit priors and parameter conversion.",
    "SimpleDSphEstimationModel composes DSphModel, FlatPriorModel and PhotometryPriorModel. "
    "args_load_data=[data] supplies a DataFrame with R_pc, vlos_kms and e_vlos_kms; "
    "kwargs_load_data may contain shared=True. Parameter vectors follow p_names_lnprob exactly. "
    "log10_ names map to 10**p; bfunc_ names map to 1-10**p.",
    "lnlikelihoods gives (N,) log densities; lnlikelihood sums them. lnpriors returns prior terms. "
    "lnposterior returns (logposterior, loglikelihood, *prior_terms) for emcee blobs. "
    "sample draws starting coordinates; sample_data simulates velocities at supplied positions.",
    "Nonempty finite 1-D data, R>0, error>=0; mean/error in km/s. "
    "The historical observation storage dtype is float32. vmem_prior_from_data defaults to False. "
    "WBIC uses inverse_temparature=1/log(N) and requires N>1. Shared data cannot be resized.",
    "Invalid prior order/schema/data raise ValueError. FittableModel requires a list args_load_data. "
    "Shared buffers must be released with release_shared_memory after workers stop.",
    CLASSICAL, NO_GRAD, "examples/docs_inference.py")
add("jeanspy.model.get_default_estimation_model", "Construct the classical convenience estimation model.",
    "data is an observed DataFrame; config is a finite ordered prior DataFrame or CSV path; "
    "photometry_prior_loc/scale specify the Gaussian in log10(re_pc).",
    "SimpleDSphEstimationModel with the standard Plummer/NFW/constant-anisotropy components.",
    "A convenience constructor does not choose scientifically justified prior bounds for the caller.",
    "Missing/invalid data or prior configuration raises; templates must be completed first.",
    CLASSICAL, NO_GRAD, "examples/docs_inference.py")

for name, purpose, params, output in [
    ("PlummerTracer", "Normalized spheroidal Plummer tracer.",
     "a_pc is equatorial scale in pc, q>0 is intrinsic axis ratio.",
     "density(R,z) is pc^-3; radial_derivative is dnu/dR in pc^-4; surface_density(x,y,inclination) "
     "is pc^-2; projected_axis_ratio returns sqrt(cos(i)^2+q^2 sin(i)^2)."),
    ("ZhaoHalo", "Spheroidal Zhao dark-matter density, force and ellipsoidal mass.",
     "rho_s (Msun/pc^3), r_s (pc), Q>0, alpha>0, beta>2, 0<=gamma<2; r_t_pc>0 "
     "is an ellipsoidal cutoff and can be infinite for the forward model.",
     "density is Msun/pc^3 and zero outside r_t_pc. potential_gradient returns "
     "(dPhi/dR,dPhi/dz) in (km/s)^2/pc, opposite gravitational acceleration. "
     "enclosed_mass(m_pc) is Msun inside the ellipsoid R^2+z^2/Q^2<=m_pc^2. J/D are scalar factors."),
    ("AxisymmetricJeans", "Direct cylindrically aligned axisymmetric Jeans solution.",
     "tracer is PlummerTracer; halo is ZhaoHalo; beta_z<1; inclination in [0,pi/2] radians; "
     "n_force, n_vertical and n_los are integer quadrature orders >=16.",
     "intrinsic_moments(R,z) returns (vR2,vz2,vphi2), each in (km/s)^2. "
     "los_second_moment(x,y) returns the surface-density-weighted second moment, not its square root."),
]:
    add("jeanspy.axisymmetric."+name, purpose, params+
        " Intrinsic R>=0 and signed z, or signed sky x/y, are finite and broadcast to one shape (pc).",
        output+" Array outputs follow the broadcast shape, including scalar output.",
        "x is the line of nodes; i=0 is face-on. No streaming prescription, stellar self-gravity, "
        "PSF or pixel average is supplied. All fixed rules require refinement.",
        "Invalid values/shapes/orders raise ValueError; wrong component types raise TypeError. "
        "Negative/nonfinite intrinsic moments raise InvalidAxisymmetricModelError. "
        "Force evaluation exactly at a cusped origin is unsupported.",
        "NumPy/SciPy CPU, frozen dataclasses.", NO_GRAD, "examples/docs_axisymmetric.py")
add("jeanspy.axisymmetric.intrinsic_axis_ratio", "Deproject an oblate projected axis ratio.",
    "q_projected in (0,1] and inclination in (0,pi/2] radians, both scalars.",
    "Scalar q=sqrt((q_projected^2-cos(i)^2)/sin(i)^2).",
    "Requires q_projected>cos(i). Face-on photometry is degenerate and rejected.",
    "Incompatible flattening, face-on angle or invalid values raise ValueError.",
    "NumPy CPU.", NO_GRAD, "intrinsic_axis_ratio(0.8, np.pi/2) gives 0.8.")
add("jeanspy.axisymmetric.InvalidAxisymmetricModelError", "Signal an inadmissible classical axisymmetric model.",
    "A normal Python exception message.", "ValueError subclass.",
    "Raised for nonfinite or negative intrinsic moments; inference rejects such proposals.",
    "This object is the exception type.", "Python host.", NO_GRAD,
    "Catch ValueError around an exploratory classical forward call; preserve failures in a benchmark.")

AXIS_PARAMS = ("params requires re_pc, rs_pc (pc), rhos_Msunpc3 (Msun/pc^3), and exactly one of "
               "q or q_projected. Optional Q, alpha, beta, gamma, beta_z and inclination "
               "(radians) have the defaults shown in the axisymmetric guide. r_t_pc "
               "is a positive ellipsoidal cutoff (pc). Use alpha/beta/gamma; spherical a/b/g names are not accepted.")
AXIS_OUTPUT = ("sigmalos2 and intrinsic_moments return (km/s)^2; density_3d is normalized pc^-3, "
               "surface_density pc^-2, mass_density_3d Msun/pc^3, enclosed_mass Msun inside "
               "an ellipsoid; potential_gradient is (km/s)^2/pc. Coordinates broadcast; "
               "intrinsic moments and forces are tuples of matching arrays.")
for name, backend, grad, errors in [
    ("jeanspy.model.AxisymmetricDSphModel", "NumPy/SciPy CPU.",
     NO_GRAD, "Invalid schema/values raise ValueError; nonphysical moments raise InvalidAxisymmetricModelError."),
    ("jeanspy.model_numpyro.AxisymmetricDSphModel", JAX,
     "Physical scalar parameters and supported coordinate values are differentiable in admissible "
     "smooth regions. Node counts, schema decisions, rejection masks and hard-cutoff boundaries "
     "are not continuous model parameters. J/D methods are not part of this JAX class.",
     "Invalid dynamic parameters yield NaN, including under jit. Schema/shape/configuration errors "
     "raise before evaluation; likelihood rejects invalid variances."),
]:
    add(name, "Functional axisymmetric model with explicit physical parameter dictionary.",
        AXIS_PARAMS+" "+SCALAR_PARAMS+" Constructor node counts n_force/n_vertical/n_los are static integers >=16.",
        AXIS_OUTPUT, "Cylindrical alignment with constant beta_z; same physical restrictions as "
        "AxisymmetricJeans. Scalar sky inputs yield scalars; centers are allowed for projected moments.",
        errors, backend, grad, "examples/docs_axisymmetric.py; examples/docs_jax.py")
add("jeanspy.model.AxisymmetricKinematicData", "Validated individual-star axisymmetric observations.",
    "x_pc, y_pc, vlos_kms and e_vlos_kms are matching nonempty finite 1-D arrays; pc and km/s. "
    "from_data accepts the supported mapping/table or an existing data object.",
    "An immutable data container; as_kwargs returns the four arrays under their public names; len is N.",
    "Signed sky coordinates include the center; velocity errors must be nonnegative.",
    "Missing fields, mismatched shapes or invalid values raise.",
    "NumPy host arrays.", NO_GRAD, "examples/docs_inference.py")
add("jeanspy.model.AxisymmetricDSphEstimationModel",
    "Classical individual-velocity inference with an axisymmetric forward model.",
    "data follows AxisymmetricKinematicData; prior is FlatPriorModel or ordered lower/upper DataFrame; "
    "fixed_params complements sampled names; dsph_model is a NumPy AxisymmetricDSphModel. "
    "p is shape (ndim,) in prior order; log10_ and bfunc_ transforms are explicit.",
    "Per-star/summed log likelihoods and prior terms. lnposterior returns posterior plus diagnostic blobs; "
    "sample(size,rng=...) generates admissible starting coordinates; sample_data(p,rng=...) gives simulated data.",
    "Gaussian LOS velocity likelihood at fixed positions; beta_z is distinct from spherical anisotropy. "
    "Exactly one of intrinsic/projected tracer flattening must be specified. Photometric prior is optional and explicit.",
    "Malformed schema/data or sampled/fixed collisions raise ValueError; inadmissible proposals "
    "give minus-infinite posterior. sample raises after max_attempts if no valid point is found.",
    CLASSICAL, NO_GRAD, "examples/docs_inference.py")

add("jeanspy.model_numpyro.Model", "Compose functional JAX model components.",
    "submodels maps the exact role names declared by the concrete class; no physical values "
    "are stored here. Numerical methods receive explicit parameters.",
    "A component container with __getitem__ role access and sampling_identity without JIT cache.",
    SCALAR_PARAMS, "Missing/extra submodel roles raise ValueError.", JAX,
    "Composition is static; differentiation applies to array-valued numerical method arguments.",
    "examples/docs_jax_spherical.py")
add("jeanspy.model_numpyro.PlummerModel", "Functional unit-integral spherical Plummer tracer.",
    "density_2d(R_pc,re_pc=...) and density_3d(r_pc,re_pc=...) use pc. "
    "sample_R(key,n,re_pc=...) takes a JAX random key and static sample count.",
    "pc^-2 or pc^-3 density with broadcast input shape; log_prob_R includes 2*pi*R "
    "and is the log radial PDF; sample_R returns shape (n,) radii in pc.",
    "Positive re_pc, nonnegative radius; log_prob_R is minus infinity outside its support.",
    "Low-level density expressions can produce NaN/inf for invalid parameters.",
    JAX, "Physical scales and valid radii are differentiable; random keys and sample counts are not.",
    "examples/docs_jax_spherical.py")
for name in ["NFWModel", "ZhaoModel"]:
    add("jeanspy.model_numpyro."+name, "Functional spherical halo density and mass.",
        "params contains rs_pc, rhos_Msunpc3 and r_t_pc; Zhao also needs a,b,g. "
        "NFW resolve_params derives and returns the mass normalization from these scale parameters. "
        "r_pc is nonnegative pc; method is auto/analytic/numeric; n_steps is a static mass quadrature order.",
        "Density Msun/pc^3; mass Msun within min(r_pc,r_t_pc), matching radius shape. "
        "enclosure_mass is an alias; valid_mass_domain returns a Boolean mask. "
        "Classical/JAX density methods themselves evaluate the untruncated profile.",
        "Positive scales/cutoff; Zhao a>0 and g<3. Finite-radius mass allows b<=3. "
        "Analytic NFW has a stable small-radius expression.",
        "Invalid physical mass proposals yield NaN. Unsupported mass-method names raise ValueError.",
        JAX, "auto chooses analytic NFW and cusp-regularized numeric Zhao. "
        "Zhao enclosed_mass_betainc/analytic does not support autodiff in shape parameters; "
        "use auto/numeric for that purpose. Hard-cutoff boundaries need separate treatment.",
        "examples/docs_jax_spherical.py")
for name in ["ConstantAnisotropyModel", "OsipkovMerrittModel", "BaesAnisotropyModel"]:
    add("jeanspy.model_numpyro."+name, "Functional spherical anisotropy and LOS kernel.",
        "params follows the corresponding classical profile names: beta_ani, or r_a, "
        "or beta_0/beta_inf/r_a/eta. Radii in pc and u=r/R dimensionless. "
        "kernel backend/order arguments are static numerical choices.",
        "Dimensionless beta/kernel and integrating factor f, with broadcast shape.",
        "Require finite beta<1, positive transition radius/sharpness where applicable. "
        "Baes large eta needs numerical refinement; the default solver chooses Abel for general Baes.",
        "Invalid static backend/options raise; invalid physical proposals can produce NaN. "
        "Warnings flag known sharp-transition limitations.",
        JAX, "Use the JAX kernel path. The explicit SciPy callback route is a reference, not "
        "a fully differentiated physical-parameter path. Verify gradients near branch and anisotropy limits.",
        "examples/docs_jax_spherical.py")
add("jeanspy.model_numpyro.DSphModel", "Functional spherical LOS second moments.",
    "Compose StellarModel, DMModel and AnisotropyModel; params supplies all physical scalars. "
    "R_pc is scalar or nonempty 1-D pc. backend is auto/kernel/abel; jit controls cached compilation. "
    "n_u/n_kernel set outer/kernel rules; n_r sets Abel grid; u_max sets radial extent; "
    "dm_mass_n_steps sets mass integration independently.",
    "Always a 1-D array of LOS variances in (km/s)^2, length one for scalar R_pc.",
    "Finite R>0. Kernel auto selects constant/Osipkov-Merritt; general Baes uses Abel. "
    "The published preliminary accuracy envelope is not a universal error bound; refine each new domain.",
    "Invalid shape/backend/options raise ValueError; invalid dynamic radii or physical values yield NaN.",
    JAX, "Physical parameters on the supported JAX mass/kernel paths; static quadrature and backend "
    "choices are not differentiated. Abel-grid boundaries and finite integration extent affect accuracy.",
    "examples/docs_jax_spherical.py")
add("jeanspy.model_numpyro.configure_runtime jeanspy.model_numpyro.get_runtime_config",
    "Configure/report effective JeansPy JAX precision and backend.",
    "configure_runtime accepts jax_enable_x64; get_runtime_config takes no arguments. "
    "Device selection should be set through environment variables before JAX initialization.",
    "Dictionary of effective runtime settings and numerical defaults.",
    "Changing precision after tracing can create a different numerical analysis and restart identity.",
    "Unsupported legacy options are rejected rather than silently changing device behavior.",
    "Host configuration of JAX.", "Configuration choices are static, not differentiable.",
    "examples/docs_jax_spherical.py")

for name, unit, restriction in [
    ("jfactor", "GeV^2 cm^-5", "gamma<1.5 (finite central annihilation integral)"),
    ("dfactor", "GeV cm^-2", "gamma<2 under the ZhaoHalo constructor domain"),
]:
    add("jeanspy.axisymmetric_factors."+name, "Integrate a circular finite-distance cone through a truncated spheroid.",
        "halo is ZhaoHalo; dist_pc is scalar observer distance (pc); roi_deg is scalar cone half-angle; "
        "inclination is radians; n_mu/n_phi/n_radial are integers >=16.",
        "Nonnegative Python float in "+unit+"; zero aperture gives zero.",
        "Require explicit finite r_t_pc, observer distance > r_t_pc*max(1,Q), 0<=roi_deg<90 and "+
        restriction+". n_phi uses a periodic rule; refine all orders.",
        "Invalid geometry/domain/order raises ValueError; wrong halo type raises TypeError.",
        "NumPy/SciPy CPU postprocessing.", NO_GRAD, "examples/docs_factors.py")
for name, description in {
    "jeanspy.model.C_J": "Unit conversion from Msun^2/pc^5 to GeV^2/cm^5.",
    "jeanspy.axisymmetric_factors.C_J": "Unit conversion from Msun^2/pc^5 to GeV^2/cm^5.",
    "jeanspy.axisymmetric_factors.C_D": "Unit conversion from Msun/pc^2 to GeV/cm^2.",
    "jeanspy.model.GMsun_m3s2": "Solar gravitational parameter G*Msun, in m^3/s^2.",
    "jeanspy.baes_eta2.DEFAULT_BAES_ETA2_KERNEL_N_QUAD": "Default static eta=2 kernel quadrature node count."
}.items():
    add(name, description, "No inputs; exported scalar constant.", description,
        "Use with the stated unit system; do not multiply a converted factor twice.",
        "No call-time validation.", "Host scalar.", "Constant.", "examples/docs_factors.py")

add("jeanspy.sampler.Sampler", "Run emcee with HDF5 storage and analysis identity checks.",
    "model supplies posterior/blobs and parameter names; p0_generator provides starting positions; "
    "nwalkers is an ensemble size; prefix sets output filename; reset=True deliberately resets "
    "the requested store; pool controls parallel evaluation. run_mcmc receives the run length/options "
    "in its real signature.",
    "HDF5 chain and diagnostic blobs; get_chain returns (draw,walker,parameter), or flattened draws. "
    "get_log_prob/get_blobs use corresponding draw/walker axes. get_dataframe returns a pandas table.",
    "Workers share a stateful model only through the supported process/storage setup. "
    "Use independent ensembles for R-hat; interacting walkers are not independent chains.",
    "Incompatible persisted analysis identity raises before continuing. "
    "Missing/malformed state, invalid parameter conversion or storage failures raise.",
    "Python/emcee host sampler around classical CPU likelihoods.", NO_GRAD,
    "examples/docs_inference.py; complete tutorials for production diagnostics.")
add("jeanspy.sampler_numpyro.ParameterSpec", "Describe one sampled coordinate and its physical transform.",
    "sample_name names a NumPyro sample site; distribution is a distribution or zero-argument factory; "
    "param_name names the physical parameter; transform is a callable; "
    "record_deterministic/deterministic_name control recorded transformed sites. "
    "exp and pow10 constructors set exponential/base-10 transforms.",
    "Specification object. build_distribution returns a NumPyro distribution; sample returns "
    "(physical_name, physical_value) and records its configured sample/deterministic sites.",
    "Priors live in the sampled coordinate. Transforming the value does not turn a log-uniform "
    "prior into a uniform prior in physical units.",
    "Invalid/duplicate names and inconsistent deterministic-site configuration raise.",
    "NumPyro/JAX.", "Transforms/distributions must support the intended JAX derivatives; "
    "discrete sample sites are not NUTS coordinates.", "examples/docs_inference.py")
add("jeanspy.sampler_numpyro.JeansLikelihoodModel jeanspy.sampler_numpyro.AxisymmetricJeansLikelihoodModel",
    "NumPyro model for independent individual stellar velocities.",
    "dsph_model is the matching JAX model; parameter_specs is a sequence of ParameterSpec; "
    "the axisymmetric subclass accepts fixed_params for complementary scalars; the spherical class "
    "uses parameter_postprocess to assemble additional fixed parameters. "
    "Velocity mean and sigmalos2 options are explicit. "
    "Calling the spherical model takes R_pc/vlos_kms/e_vlos_kms; the axisymmetric model takes x_pc/y_pc "
    "instead of R_pc. Observation arrays are matching finite nonempty 1-D arrays.",
    "__call__ returns None while registering NumPyro sample, deterministic and likelihood sites; "
    "sample_parameters returns the transformed parameter mapping. Variance includes measurement error squared.",
    "Positions in pc and velocities/errors in km/s, errors>=0. Gaussian LOS closure at fixed positions. "
    "The standard class does not add membership mixtures, velocity-cut normalization or binaries.",
    "Bad schema/shape and sampled/fixed collisions raise; inadmissible forward variances are rejected "
    "with minus-infinite density.",
    "NumPyro with JAX forward model.", "Supported continuous physical parameters are traceable. "
    "J/D factors are not likelihood sites. Verify derivatives for custom prior transforms.",
    "examples/docs_numpyro_inference.py")
add("jeanspy.sampler_numpyro.NumPyroSampler",
    "Persist and resume an explicit NumPyro MCMC analysis.",
    "mcmc is numpyro.infer.MCMC; output_dir is a filesystem path; storage_backend chooses "
    "zarr/netcdf4/h5netcdf; async_writes toggles the writer; arviz_converter optionally returns xarray.DataTree. "
    "run(rng_key,*args,**kwargs) forwards data to the model, with explicit resume/save/write flags.",
    "run returns SamplerRunResult. Samples use ArviZ chain/draw dimensions; load_samples(combine=True) "
    "returns a combined DataTree, False a list. save_samples_chunk returns (index,path,submitted). "
    "save_checkpoint/load_checkpoint return paths; flush/close return None after awaiting writes.",
    "Use a context manager or close/flush before relying on completed files. "
    "resume='auto' checks available state; resume=True requires valid state. "
    "Metadata binds data, prior, model/source/dependencies and effective runtime configuration. "
    "A checkpoint contains trusted Python serialization; open only your trusted analysis output.",
    "Missing/inconsistent analysis metadata or checkpoint identity raises before restarting. "
    "Write failures propagate from flush/close. Use a new directory for a changed analysis.",
    "NumPyro/JAX inference, host-side ArviZ storage.", "The model may be differentiated; "
    "sampling control, disk I/O and checkpoint state are not differentiable.",
    "examples/docs_numpyro_inference.py")
add("jeanspy.sampler_numpyro.SamplerRunResult", "Report one NumPyroSampler run and persistence request.",
    "resumed and write_submitted are booleans; checkpoint_path/chunk_path are paths or None; "
    "chunk_index is an integer or None.",
    "Frozen result record. write_submitted does not by itself confirm an asynchronous write finished.",
    "Inspect returned paths after flush/close before claiming durable completion.",
    "No numerical-domain validation.", "Python metadata.", NO_GRAD, "examples/docs_inference.py")
add("jeanspy.sampler_numpyro.StorageBackend", "Allowed sample-store backend names.",
    "Literal storage selector.", "One of zarr, netcdf4 or h5netcdf.",
    "Install the corresponding optional storage dependencies.", "Unsupported names raise ValueError in sampler construction.",
    "Host I/O.", NO_GRAD, "examples/docs_inference.py")

add("jeanspy.dequad.hashable", "Test whether Python can hash an object.",
    "x is any Python object.", "bool; catches TypeError from hash(x).",
    "Hashability is not a check of array content identity.", "Exceptions other than TypeError propagate.",
    "Python host.", NO_GRAD, "hashable((1,2)) is True; hashable(np.array([1])) is False.")
add("jeanspy.dequad.memorize", "Cache a callable for hashable argument tuples.",
    "callable is a Python function; the wrapper accepts its positional/keyword arguments.",
    "Wrapped callable with an in-memory cache; unhashable arguments bypass caching.",
    "Pure functions only; key order follows supplied kwargs. Mutable returned values are shared cached objects.",
    "Original callable exceptions propagate.", "Python host.", NO_GRAD,
    "Used internally for quadrature nodes; use dequad for numerical integration.")
add("jeanspy.dequad.generate_x_w", "Construct fixed double-exponential integration nodes and weights.",
    "Call generate_x_w(a,b,n,xp=np): scalar limits, integer n>=2, NumPy-like namespace xp.",
    "Pair (x,w) with shape (n,) for scalar bounds; weights include the transformation and step.",
    "Supported intervals are finite (a,b), finite a to +inf, or the full infinite line. "
    "Mixed vector-bound interval types are not a supported public contract.",
    "Invalid interval combinations/orders are not uniformly checked; use only the supported cases.",
    "NumPy node selection; xp does not make this a fully traced JAX routine.", NO_GRAD,
    "examples/docs_numerics.py")
add("jeanspy.dequad.dequad", "Fixed double-exponential quadrature.",
    "See the full argument contract below; scalar limits and vectorized integrand.",
    "Integral with the requested axis removed; no error estimate.",
    "Refine n to verify convergence; nonfinite replacement can discard failures.",
    "Nonfinite weighted values issue warnings unless the explicit replacement option is enabled.",
    "NumPy CPU; xp is not an end-to-end JAX contract.", NO_GRAD, "examples/docs_numerics.py")

for name in ["GaussLegendre01","TanhSinh01"]:
    add("jeanspy.hyp2f1_jax."+name, "Store a fixed quadrature rule on the unit interval.",
        "x and w are matching JAX arrays of nodes and weights, normally shape (n,).",
        "Frozen dataclass with x/w fields; no automatic node generation occurs in this constructor.",
        "Nodes should lie in [0,1] and weights correspond to the chosen integration rule.",
        "The container itself does not validate the mathematical rule.", JAX,
        "Nodes are treated as constants by normal solver use.", "examples/docs_numerics.py")
for name in ["hyp2f1_1b_d_series","hyp2f1_1b_3half_series","hyp2f1_1b_3half_quad",
             "hyp2f1_1b_3half_asymptotic","hyp2f1_1b_3half"]:
    add("jeanspy.hyp2f1_jax."+name, "Specialized real hypergeometric evaluation for Jeans kernels.",
        "b,w are dimensionless scalars/broadcastable JAX arrays; w is in [0,1). "
        "The general series also takes d. n_terms/n_points/quad_rule and thresholds are static controls "
        "shown in the signature.",
        "Array approximation to 2F1(1,b;3/2;w), or 2F1(1,b;d;w) for the general series.",
        "The Euler integral helper requires 0<b<3/2; d cannot cross denominator poles. "
        "Fixed series loses efficiency near w=1; asymptotic formulas have restricted regimes. "
        "Use the combined dispatcher or inspect convergence, rather than extending a specialized formula.",
        "Unsupported rule choices raise ValueError; out-of-domain numerical values can become NaN/inf.",
        JAX, "Fixed-loop continuous expressions support autodiff in their valid regions; "
        "piecewise thresholds and parameter singularities need checks.", "examples/docs_numerics.py")
add("jeanspy.baes_eta2.BaesEta2AnisotropyModel", "Fixed-eta=2 Baes anisotropy for the JAX kernel route.",
    "Physical params beta_0,beta_inf and positive r_a (pc); eta is fixed at two. "
    "beta/f use radius in pc; kernel uses dimensionless u=r/R and projected radius in pc.",
    "Dimensionless anisotropy/kernel and integrating factor f.",
    "This specialized model is not the arbitrary-eta Baes model. Check the stated supported prior envelope.",
    "Invalid proposals may yield nonfinite values; downstream likelihood rejects invalid forward variance.",
    JAX, "Continuous beta_0/beta_inf/r_a derivatives on the supported fixed-rule path.",
    "examples/docs_jax_spherical.py")
add("jeanspy.baes_eta2.baes_eta2_kernel_jax",
    "Fixed-rule eta=2 Baes projection kernel.",
    "u>=1 and R_pc>0, beta_0/beta_inf dimensionless, r_a>0 in pc, n_kernel static. "
    "See the exact signature for supported quadrature options.",
    "Dimensionless kernel with broadcast array shape.",
    "Use the documented eta=2 domain and refine quadrature for extreme anisotropies or scales.",
    "This low-level helper clamps radii/arguments and replaces nonfinite kernel values; "
    "a finite output is not evidence of valid inputs. Validate the physical domain before calling it.",
    JAX, "Differentiable continuous physical parameters in the supported regime.",
    "examples/docs_numerics.py")
add("jeanspy.baes_eta2.baes_eta2_kernel_appell_reference",
    "High-precision Appell-function reference for the eta=2 kernel.",
    "Broadcastable u/R_pc arrays and scalar beta_0/beta_inf/r_a with the units/domain of the JAX eta=2 kernel; "
    "dps sets mpmath working precision.",
    "NumPy dimensionless kernel array with the broadcast u/R_pc shape.",
    "Independent numerical check; not the production JAX inference route.",
    "mpmath evaluation/convergence exceptions can propagate.",
    "Python/mpmath CPU.", NO_GRAD, "examples/docs_numerics.py")


CONTRACTS["jeanspy.model.SersicModel"]["Domain"] += (
    " The bundled b_n interpolator covers n from about 0.02 to 15.17; numerical "
    "deprojection does not remove that table limit. VM20 and VM20bis have the "
    "stricter n/r domains specified by their methods."
)

from api_method_contracts import extend_methods
extend_methods(CONTRACTS)


def append_contract(app, what, name, obj, options, lines):
    """Append reviewed contracts to canonical autodoc objects and their aliases."""
    item = CONTRACTS.get(name)
    if item is None:
        return
    lines += ["", ".. rubric:: Usage contract", ""]
    for label, value in item.items():
        # Contracts are plain prose, not reStructuredText markup. Literal
        # transform prefixes and unpacking operators must remain literal.
        escaped = value.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_")
        lines += [f"**{label}.** {escaped}", ""]


def verify_inventory(inventory):
    missing = [item["canonical"] for item in inventory
               if item["canonical"] not in CONTRACTS]
    if missing:
        raise ValueError("Missing public API usage contracts: " + ", ".join(sorted(set(missing))))
