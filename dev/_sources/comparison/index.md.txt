# Related software and comparison questions

Source review date: **2026-09-13**. Not a ranking. Absent/unlocated means not established in the cited inspected sources. Survey pins and measurement pins are separate. Differentiation refers to physical model parameters unless stated otherwise.

This survey has not established a speed or inference-quality ranking. Published performance examples use different workloads and hardware; they are not substituted for measurements in this project.

The working hypothesis is that gradients of direct spherical and axisymmetric Jeans models can help inference while retaining numerical accuracy. SKiNN already provides differentiable emulation, GLaD already uses JAX/GPU Jeans calculations, and JamPy 9 introduces a substantially different spectral algorithm. None permits a novelty claim based on differentiability or GPU support alone. CJAM and later discrete JAM studies also establish individual stellar velocities and contamination models as existing methods.

## Physical models and observations

| Code | Target | Geometry | Physical assumptions | Observables / likelihood | Model freedom |
| --- | --- | --- | --- | --- | --- |
| JeansPy [sources](#software-jeanspy) | Resolved stellar kinematics of dSphs | Spherical; cylindrically aligned axisymmetric | Equilibrium second-order Jeans; tracer and halo components; spherical anisotropy profiles or constant cylindrical beta_z; current axisymmetric tracer has no self-gravity. | Projected LOS second moments; Gaussian individual-velocity likelihood with measurement errors. | Spherical Plummer and classical additional tracers, NFW/Zhao halo and several anisotropy laws; axisymmetric Plummer + Zhao with flattening and inclination. |
| JamPy 9.0.2 (current JAM) [sources](#software-jampy-9-0-2-current-jam) | Integrated-light galaxy and stellar-system kinematics | Axisymmetric public solver uses spherical alignment | Spectral solution of the two-dimensional Jeans boundary problem with general beta(r, theta). The public cylindrical-alignment solver was removed in v9. | Intrinsic/projected first and second moments, including PMs, PSF and aperture treatment. | Projected tracer requires MGE; potential may be MGE or a callable. Intrinsic solver accepts general density/potential callables and anisotropy. |
| JamPy 8.1.4 (legacy reference) [sources](#software-jampy-8-1-4-legacy-reference) | Integrated-light galaxy and stellar-system kinematics | Spherical and axisymmetric; cylindrical or spherical alignment | Second-order Jeans with multi-Gaussian expansions (MGE), anisotropy, inclination and optional rotation prescriptions. | Projected first/second velocity moments including proper motions; PSF and aperture treatment. | Flexible luminous/mass MGE components, M/L, black-hole mass and anisotropy. |
| CJAM [sources](#software-cjam) | Discrete stellar kinematics; original application to omega Centauri | Axisymmetric, cylindrical JAM closure | MGE tracer and mass; component anisotropy and rotation; inclination. | Three first and six second velocity moments; the paper uses individual LOS/PM data with a contaminating population. | Separate tracer/mass MGEs, component mass scalings, anisotropy, rotation and optional black hole. |
| GLaD [sources](#software-glad) | Joint strong-lens imaging and spatially resolved stellar dynamics | Axisymmetric, spherically aligned JAM-based dynamics | Composite lens mass and light models, MGE and spherical alignment; lensing and dynamics share mass parameters. | Lensed image information and two-dimensional stellar second moments; paper combines several likelihood terms. | Flexible composite lens models, stellar/DM components and inclination; not the same tracer/closure as JeansPy's cylindrical model. |
| SKiNN [sources](#software-skinn) | Stellar kinematic maps of strong-lens early-type galaxies | Axisymmetric cylindrical JAM training targets | Neural approximation to JAM maps for elliptical power-law total mass and Sersic light within a specified training distribution. | Spatially resolved second-moment maps; intended for dynamical/lensing analyses. | Eight inputs in the published model: lens/mass scale and slope/flattening, light scale/index/flattening, anisotropy and inclination. |
| AGAMA [sources](#software-agama) | General galactic dynamics and galaxy modeling | Spherical, axisymmetric and more general potentials/orbit models | Potentials, forces, actions, distribution functions, DF moments, self-consistent and Schwarzschild models. Axisymmetric Jeans models are explicitly recorded in the May 2018 changelog. | Positions/velocities, intrinsic/projected moments and model-dependent observables. | Broad potential, density and DF libraries; beyond a single Jeans closure. |
| galpy [sources](#software-galpy) | General Galactic dynamics, potentials, orbits and DFs | General dynamics library; its jeans.py helpers assume spherical symmetry | Spherical Jeans helper accepts tracer density and constant or radial anisotropy; sphericity of the supplied potential is not checked. | sigmar and sigmalos return velocity dispersions; broader library supplies other dynamical observables. | Composable potentials, user-supplied density and anisotropy functions. |
| GravSphere1 / binulator [sources](#software-gravsphere1-binulator) | Spherical stellar systems, especially dSphs | Spherical equilibrium | Second-order Jeans with virial shape parameters (fourth-moment information), optional proper motions; current binulator release differs from the original free-form implementation. | Binned photometry/velocity moments, virial shape constraints and optional PMs. | Flexible halo/anisotropy and tracer modeling; current public model choices must be specified. |
| GravSphere2 [sources](#software-gravsphere2) | Spherical stellar systems from clusters to galaxies | Spherical equilibrium | Second- and fourth-order Jeans equations with flexible anisotropy at both orders and non-Gaussian velocity PDF modeling. | Individual LOS velocities and optional proper motions; photometric tracers may be unbinned. | Flexible mass/anisotropy/tracer models and higher-moment parameters. |
| pyGravSphere [sources](#software-pygravsphere) | dSph mass profiles | Spherical equilibrium | Free-form density and anisotropy Jeans analysis with binned kinematics and virial shape information. | Photometric/kinematic star inputs are preprocessed into moment constraints. | Nonparametric density description and flexible anisotropy/tracer parameters. |
| MAMPOSSt [sources](#software-mamposst) | Galaxy clusters/groups, also spherical galaxies/dSphs | Spherical equilibrium | Parametric mass/anisotropy profiles and a chosen 3D velocity distribution (Gaussian in the founding paper). LOS projection generally gives a non-Gaussian velocity distribution. | Unbinned projected phase space (R, v_los), including its radial selection. | Mass/tracer scale radii and anisotropy parameters; multiple tracer populations in public example. |
| CLUMPY [sources](#software-clumpy) | Dark-matter annihilation/decay signals; dSph Jeans analysis | Spherical Jeans module; signal calculations also allow nonspherical halos/substructure | Parametric Jeans mass modeling plus J/D integrals and particle/signal calculations; signal geometry is broader than Jeans geometry. | Binned moments or individual velocities for Jeans fitting; J/D factors, fluxes and skymaps. | Selectable density/light/anisotropy models and halo/subhalo prescriptions. |
| Gala [sources](#software-gala) | General Galactic and gravitational dynamics | Potential/orbit-dependent; not limited to spherical Jeans models | Potential/force models, Hamiltonian orbits, coordinates and dynamical analysis. | Trajectories and dynamical quantities; no complete matching dSph Jeans likelihood established in inspected API. | Composable potentials and flexible phase-space initial conditions. |
| galax [sources](#software-galax) | General Galactic/gravitational dynamics in JAX | Potential/orbit-dependent | Unit-aware gravitational potentials, phase-space coordinates and orbit integration. | Orbits and potential-derived quantities; a matching dSph Jeans likelihood was not located in the inspected surface. | Composable potential and coordinate models. |

## Numerics and inference

| Code | Numerical method | Differentiation | CPU / GPU | Inference |
| --- | --- | --- | --- | --- |
| JeansPy [sources](#software-jeanspy) | Direct fixed quadrature (and classical adaptive reference routes); no trained surrogate. | Physical-parameter autodiff through supported JAX forward/likelihood paths. NumPy J/D factors are postprocessing; analytic Zhao incomplete-beta shape derivatives are unsupported. | NumPy/SciPy CPU; JAX CPU/GPU, precision explicitly configured. | emcee and NumPyro wrappers; persisted analysis identity, samples and restart. |
| JamPy 9.0.2 (current JAM) [sources](#software-jampy-9-0-2-current-jam) | Direct spectral method in log radius, finite domain with outer Robin condition; line-of-sight integration remains for projection. | spectral_derivs computes spatial density/potential derivatives, not automatic derivatives with respect to physical model parameters. Inspected implementation uses NumPy/SciPy. | CPU release; official changelog says a GPU-optimized implementation is in development, not available as this measured release. | Forward moments and fitting interface; probabilistic sampling is composed externally. |
| JamPy 8.1.4 (legacy reference) [sources](#software-jampy-8-1-4-legacy-reference) | Direct integration of MGE Jeans expressions. | No general physical-parameter autodiff interface documented in the inspected Python package. | Python/NumPy/SciPy CPU in the inspected implementation. | Built-in fitting options; external Bayesian inference can wrap the forward model. |
| CJAM [sources](#software-cjam) | Direct C/GSL integrations with an optional Python/Cython wrapper and interpolation grid. | No physical-parameter automatic differentiation interface in the documented C implementation. | CPU C/GSL. | Discrete likelihood framework in Watkins et al.; the public repository supplies the moment engine. |
| GLaD [sources](#software-glad) | Direct JAX rewrite of part of JAM with fixed integration grids; no learned surrogate for the dynamical solver described in the paper. | JAX implementation is explicit. End-to-end physical-parameter gradient accuracy across the complete lensing/inference pipeline is not established by the inspected paper. | CPU/GPU hybrid; JAX matrix/integration work on GPU. | Joint Bayesian modeling using emcee in the paper. |
| SKiNN [sources](#software-skinn) | Trained neural surrogate; training and weight provenance are part of the method. | Differentiable learned mapping; existing differentiability precludes a novelty claim based only on autodiff. | GPU inference/training described; current README requires CUDA. | Provides surrogate predictions and parameter sensitivities; compare the complete chosen external inference workflow explicitly. |
| AGAMA [sources](#software-agama) | Direct C++ numerical dynamics with Python/Fortran interfaces. | Spatial force/derivative interfaces exist; these are distinct from an end-to-end physical-parameter autodiff likelihood. | CPU C++ with parallel capabilities; no matching GPU Jeans path established here. | Fitting framework and examples; inference target depends on the selected DF/model. |
| galpy [sources](#software-galpy) | Direct NumPy/SciPy Jeans integration; broader package includes C acceleration. Current source adds interpolation/refinement for radial moments. | No JAX physical-parameter autodiff in the inspected Jeans helper. | CPU Python/C, depending on component; inspected Jeans integrations use SciPy. | Forward helpers with external inference; general library also has fitting utilities. |
| GravSphere1 / binulator [sources](#software-gravsphere1-binulator) | Direct NumPy/SciPy integrations and preprocessing. | No physical-parameter autodiff interface documented in inspected code. | CPU; parallel sampling available. | binulator preprocessing then emcee-based mass modeling. |
| GravSphere2 [sources](#software-gravsphere2) | Direct higher-order numerical integration and velocity-PDF construction. | No physical-parameter autodiff interface documented in inspected implementation. | CPU Python/NumPy/SciPy; multiprocessing examples. | dynesty nested sampling and checkpointing. |
| pyGravSphere [sources](#software-pygravsphere) | Direct compiled extension via SWIG and Python orchestration. | No physical-parameter autodiff interface documented. | CPU compiled extension, optional multiprocessing/MPI. | emcee wrapper with preprocessing and stored chains. |
| MAMPOSSt [sources](#software-mamposst) | Direct integration of a projected phase-space model. | No end-to-end physical-parameter autodiff documented in inspected repository. | CPU native code; build instructions use INSTALL.csh. | Maximum-likelihood method in paper; repository includes MCMC output/test workflow. |
| CLUMPY [sources](#software-clumpy) | Direct C/C++ integration and statistical workflows. | No physical-parameter autodiff interface documented in the inspected Jeans module. | CPU C/C++; dependencies vary by release and optional modules. | jeansChi2 minimization/bootstrap and jeansMCMC (GreAT). |
| Gala [sources](#software-gala) | Direct Python/C numerical dynamics. | Spatial potential derivatives are supported; this does not establish JAX parameter gradients for a Jeans inference pipeline. | CPU Python/C implementation described by official documentation. | Building blocks and examples for externally composed analyses. |
| galax [sources](#software-galax) | Direct JAX dynamics; no learned surrogate implied by JAX. | Autodiff is an explicit project feature; comparison must distinguish its potential/orbit derivatives from the Jeans forward/likelihood tested here. | JAX CPU/GPU capability; actual device and precision require measurement. | Composable differentiable primitives; no complete matching Jeans sampler established here. |

## Versions and availability

| Code | Public availability | Code surveyed / measured | Paper |
| --- | --- | --- | --- |
| JeansPy [sources](#software-jeanspy) | Public BSD-3-Clause repository; alpha package and development source differ. | Baseline 1a0ad4028d26af1df389ebdfdf992285ec50f8bb | This work; no accepted-paper claim |
| JamPy 9.0.2 (current JAM) [sources](#software-jampy-9-0-2-current-jam) | PyPI source/wheel under proprietary noncommercial terms; redistribution is prohibited. | 9.0.2, uploaded 2026-07-01; wheel SHA256 1e35aef738979b1edd24aacc25d3208faa26ed96822361dd7f58ef4be663d919 | Cappellari 2026, MNRAS 549, stag420; arXiv:2601.16179v2 |
| JamPy 8.1.4 (legacy reference) [sources](#software-jampy-8-1-4-legacy-reference) | Source distributed through PyPI; package declares a proprietary license. Do not equate downloadability with an open-source license. | jampy 8.1.4; historical quadrature reference retained with original MGE coefficients | Cappellari 2008, MNRAS 390, 71; Cappellari 2020, MNRAS 494, 4819 |
| CJAM [sources](#software-cjam) | Public BSD-2-Clause source. | 9394ac44572d3dafc9973f90e59fb89d70f3694c (2022-08-03) | Watkins et al. 2013, MNRAS 436, 2598; arXiv:1308.4789 |
| GLaD [sources](#software-glad) | Paper publicly accessible; no public code repository was located in the inspected full text and cited software links. This is an access finding, not proof of non-public code. | No independently accessible GLaD code pin located | Wang et al. 2025, A&A 701, A280; arXiv:2504.01302v2 |
| SKiNN [sources](#software-skinn) | Public code and separately downloadable weights; no license detected in GitHub metadata. Weight terms must be checked separately. | d941b6a049a40a6e1e901c2f221f08ac0b72f4d5; README announces updated weights on 2026-01-15, DOI 10.58119/ULG/WZFXYD | A&A 2023 paper, DOI 10.1051/0004-6361/202347507; arXiv:2307.10381 |
| AGAMA [sources](#software-agama) | Public source; inspect the repository license terms rather than interpreting GitHub's NOASSERTION as a license. | Survey f302756b8af2b763db58e278e30478517dc8eea3; existing local submodule 4e62db6 (full pin in baseline git tree) | Vasiliev 2019, MNRAS 482, 1525 |
| galpy [sources](#software-galpy) | Public BSD-3-Clause source. | Survey d243f989b61ec6398f46fafa49b886feac1134bf; pin the installed release separately for each measurement. | Bovy 2015, ApJS 216, 29; arXiv:1412.3451 |
| GravSphere1 / binulator [sources](#software-gravsphere1-binulator) | Public repository; no license detected in GitHub metadata. | 09fa555aa7842aed3ba139e978c1e20d6d00e2d8 | Read & Steger 2017, MNRAS 471, 4541; additional current-code citations listed in README |
| GravSphere2 [sources](#software-gravsphere2) | Public MIT repository. | Survey de56ee61b377608e63d4420f424def7ceaec942f; existing submodule 06d6a507 (full pin in baseline git tree) | Banares-Hernandez, Read & Julio 2026, A&A 705, A212; arXiv:2509.24103v3 corrects equations 17, 18, 20, 23 |
| pyGravSphere [sources](#software-pygravsphere) | Public GPL-3.0 repository; README documents legacy dependency versions. | fe811462da0201bb6778d8636a8e497413dc50c2 | Original GravSphere family; Genina et al. 2020, MNRAS 498, 144, plus relevant code-use citations |
| MAMPOSSt [sources](#software-mamposst) | Public GPL-3.0-or-later GitLab repository; no tags listed. | b5148fc6412d7691e642c8e815ec69b3f575edc5 | Mamon, Biviano & Boue 2013, MNRAS 429, 3079; arXiv:1212.1455v2 |
| CLUMPY [sources](#software-clumpy) | Public GitLab code; release-specific license/dependency terms apply. | Documentation v3.1.1; surveyed master 101aff8ecd2da20a71f7eb83fb6be2f8d4e99679; README warns master installation differs from release. | Bonnivard et al. 2016 (Jeans module); Hutten, Combet & Maurin 2019, CPC 235, 336 (v3) |
| Gala [sources](#software-gala) | Public MIT source. | Survey b87d6b553de628ff53ea25871f1b2352505a5cad | Price-Whelan 2017, JOSS 2(18), 388 |
| galax [sources](#software-galax) | Public MIT source; README says documentation is forthcoming. | 192b2b4d1bf414fe2ed53866e5d248a0f048b916 | Repository citation; no primary methods paper located in README |

## Comparison boundaries and sources

(software-jeanspy)=
### JeansPy

Matched spherical and axisymmetric forward/likelihood tests; native examples and conditional coverage separately.

[baseline source](https://github.com/gomeshun/jeanspy/tree/1a0ad4028d26af1df389ebdfdf992285ec50f8bb).

(software-jampy-9-0-2-current-jam)=
### JamPy 9.0.2 (current JAM)

The frozen Plummer tests use beta=beta_z=0, where the alignment distinction vanishes. Both fine-grid cases pass the declared accuracy gates; the coarse 45x9 intrinsic grid returns NaNs and is retained. Nonzero cylindrical anisotropy and JamPy's suggested approximation are different physical models. See the current JAM validation page for results and MGE limits.

[JamPy 9.0.2 API and changelog](https://pypi.org/project/jampy/9.0.2/); [spectral Jeans paper](https://arxiv.org/abs/2601.16179v2).

(software-jampy-8-1-4-legacy-reference)=
### JamPy 8.1.4 (legacy reference)

Cylindrical alignment with the same target profiles approximated by MGE; report MGE residuals and fitting cost. Spherical-aligned JAM is a different closure.

[JAM 2008](https://arxiv.org/abs/0806.0042); [JAM 2020](https://arxiv.org/abs/1907.09894); [JamPy 8.1.4](https://pypi.org/project/jampy/8.1.4/).

(software-cjam)=
### CJAM

Discrete stellar velocities, anisotropy, inclination and contamination are established precedents. They cannot individually establish JeansPy novelty. No new CJAM runtime measurement is reported.

[public code and documentation](https://github.com/lauralwatkins/cjam/tree/9394ac44572d3dafc9973f90e59fb89d70f3694c); [discrete dynamical models](https://arxiv.org/abs/1308.4789).

(software-glad)=
### GLaD

Method/workflow comparison; not a numerically matched cylindrical Plummer+Zhao benchmark.

[GLaD paper](https://arxiv.org/abs/2504.01302); [published GLaD](https://doi.org/10.1051/0004-6361/202554861).

(software-skinn)=
### SKiNN

Native in-domain model only. Plummer+Zhao dSphs do not match the published training family; no out-of-domain speed/accuracy claim.

[SKiNN paper](https://arxiv.org/abs/2307.10381); [pinned code and weight instructions](https://github.com/mattgomer/SKiNN/tree/d941b6a049a40a6e1e901c2f221f08ac0b72f4d5).

(software-agama)=
### AGAMA

Independent potential/DF-moment reference when assumptions agree. Rebuilding a DF each proposal measures that workflow, not a generic Jeans-solver limit.

[AGAMA paper](https://academic.oup.com/mnras/article/482/2/1525/5114593); [surveyed source](https://github.com/GalacticDynamics-Oxford/Agama/tree/f302756b8af2b763db58e278e30478517dc8eea3); [AGAMA change history](https://github.com/GalacticDynamics-Oxford/Agama/blob/f302756b8af2b763db58e278e30478517dc8eea3/NEWS).

(software-galpy)=
### galpy

Matched spherical density/potential/beta predictions after explicit unit conversion; fresh measurements needed for current optimized helper.

[galpy paper](https://arxiv.org/abs/1412.3451); [inspected Jeans source](https://github.com/jobovy/galpy/blob/d243f989b61ec6398f46fafa49b886feac1134bf/galpy/df/jeans.py).

(software-gravsphere1-binulator)=
### GravSphere1 / binulator

Common second-moment forward subset versus native binned/VSP workflow separately; include preprocessing in native timing.

[GravSphere paper](https://academic.oup.com/mnras/article/471/4/4541/3979468); [pinned binulator release](https://github.com/justinread/gravsphere/tree/09fa555aa7842aed3ba139e978c1e20d6d00e2d8).

(software-gravsphere2)=
### GravSphere2

Second-moment subroutine only if inputs/definitions match; native non-Gaussian inference cannot be equated to JeansPy's Gaussian likelihood.

[GravSphere2 v3 paper](https://arxiv.org/abs/2509.24103v3); [surveyed code](https://github.com/dadams42/GravSphere2/tree/de56ee61b377608e63d4420f424def7ceaec942f).

(software-pygravsphere)=
### pyGravSphere

Native free-form/VSP workflow is scientifically distinct. Previous installation failure is preliminary; any new failure requires its own log and environment.

[pinned source and build instructions](https://github.com/AnnaGenina/pyGravSphere/tree/fe811462da0201bb6778d8636a8e497413dc50c2); [GravSphere foundation](https://academic.oup.com/mnras/article/471/4/4541/3979468).

(software-mamposst)=
### MAMPOSSt

Native phase-space likelihood comparison; not a matched Gaussian LOS likelihood without an explicit mathematical reduction.

[MAMPOSSt paper](https://arxiv.org/abs/1212.1455v2); [public source](https://gitlab.com/gmamon/MAMPOSSt/-/tree/b5148fc6412d7691e642c8e815ec69b3f575edc5).

(software-clumpy)=
### CLUMPY

Matched spherical moments or J/D geometry after explicit cuts/units; native all-sky/substructure signal work is outside JeansPy's scope.

[CLUMPY v3](https://arxiv.org/abs/1806.08639v2); [Jeans module paper](https://arxiv.org/abs/1506.07628); [v3.1.1 Jeans documentation](https://clumpy.gitlab.io/CLUMPY/v3.1.1/doc_modules_jeans.html); [surveyed code](https://gitlab.com/clumpy/CLUMPY/-/tree/101aff8ecd2da20a71f7eb83fb6be2f8d4e99679).

(software-gala)=
### Gala

Scope and interoperability comparison; no invented Jeans timing or inference score.

[Gala paper](https://joss.theoj.org/papers/10.21105/joss.00388); [official documentation](https://gala.adrian.pw/en/latest/); [surveyed source](https://github.com/adrn/gala/tree/b87d6b553de628ff53ea25871f1b2352505a5cad).

(software-galax)=
### galax

Scope and autodiff-context comparison; no fabricated Jeans benchmark.

[surveyed source and documented features](https://github.com/GalacticDynamics/galax/tree/192b2b4d1bf414fe2ed53866e5d248a0f048b916).

The [current JAM validation](../validation/jam9.md) records the selected
isotropic accuracy tests, their failed coarse setting and their limits.

## Claims requiring new measurements

1. Accuracy and physical-parameter gradients over a declared parameter domain.
2. Prediction and gradient cost including preparation, compilation and transfers.
3. Agreement of posterior summaries under identical priors and likelihoods.
4. Time to a predeclared inference-precision target, counting failed runs.
5. Mock recovery and interval coverage with simulation uncertainty.

Full-model calibration, general-Hessian accuracy, membership mixtures and variational inference are not claimed merely because JAX or NumPyro can represent them. Current NumPy J/D integration remains separate from the differentiated kinematic likelihood.

The machine-readable source is {download}`comparison.json <../../../validation/release/comparison.json>`. Regenerate this page with `python scripts/generate_comparison_docs.py`.
