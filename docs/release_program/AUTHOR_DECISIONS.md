# Author decisions before production experiments

Prepared 2026-09-12, before new production inference or calibration runs.
These are proposals, not recorded author approvals. Documentation, data-format
inspection, source research and implementation smoke checks can proceed.

## Publication venue

Proposed order for the full methods paper: **A&A, MNRAS, CPC, ApJS, JOSS**.

| Rank | Venue | Fit and concrete preparation requirement |
| --- | --- | --- |
| 1 | A&A | Numerical methods and codes section; recent SKiNN, GLaD and GravSphere2 precedents directly overlap this readership. Prepare an A&A LaTeX manuscript with a structured abstract, independent validation, explicit limitations and reproducibility materials. The journal's current full author-guide download returned HTTP 403 here; indexed official section information and published precedents were accessible. Exact formatting details will be verified before submission packaging. |
| 2 | MNRAS | Strong Jeans/galaxy-dynamics readership and JAM, AGAMA and GravSphere precedents. A scientific methods paper with a substantive validation result fits better than an API description alone. A Data Availability statement is required in the end matter. |
| 3 | CPC | Strong fit if numerical algorithms, implementation and reproducible performance are the main contribution; CLUMPY v3 is a relevant precedent. A code distribution/program description is central. The live full guide was inaccessible (403), so its exact current package requirements need a final check before formatting. |
| 4 | ApJS | Appropriate for an extensive reference resource with significant new astronomy/astrophysics research; galpy is a precedent. The current AAS scope also explicitly associates astronomical software/computing with AJ, so ApJS fit should be justified by the breadth and reference value of this paper rather than galpy alone. |
| 5 | JOSS | Strong software-review option or companion paper, but the short software-paper format cannot carry the planned full numerical/scientific argument. Current screening requires sustained public development, demonstrated research use and open-source practices. AI assistance must be disclosed and human review asserted by the authors after that review actually occurs. |

Primary sources checked:

- [A&A author guide, indexed section list](https://www.aanda.org/doc_journal/instructions/aadoc.pdf).
- [MNRAS instructions](https://academic.oup.com/mnras/pages/General_Instructions).
- [AAS scope statements](https://journals.aas.org/scope-statements/).
- [CPC author guide](https://www.sciencedirect.com/journal/computer-physics-communications/publish/guide-for-authors) (access failure retained).
- [JOSS submission and screening requirements](https://joss.readthedocs.io/en/latest/submitting.html).
- [SKiNN](https://arxiv.org/abs/2307.10381), [GLaD](https://arxiv.org/abs/2504.01302),
  [GravSphere2](https://arxiv.org/abs/2509.24103), [CLUMPY v3](https://arxiv.org/abs/1806.08639),
  [galpy](https://arxiv.org/abs/1412.3451), [Gala](https://joss.theoj.org/papers/10.21105/joss.00388).

The author must choose the venue. No submission, payment or acceptance of
journal terms is part of this work.

## Compute envelope

Proposed initial production envelope: **24 hours of elapsed scheduled compute,
at most 8 CPU threads or one RTX 3090, and 16 GiB per experiment process**.
No rented resources. Within it, record a separate bound and deterministic
stop rule for every experiment before launch. A budget failure is a reportable
result; do not silently extend runs, lower diagnostics or replace seeds.

The protocol will include CPU float64 accuracy/gradients, CPU float32 and actual
GPU measurements, matched spherical and axisymmetric likelihood comparisons,
joint-parameter synthetic fits and conditional repeated-mock recovery/coverage.
Conditional coverage with fixed nuisance parameters will be labeled as such,
and will not be used to claim full-model calibration. If the broader joint
experiment cannot meet its criteria within the envelope, report that limit.

## Draco scientific recipe

Proposed default tutorial: spherical Plummer + NFW with constant spherical
anisotropy, and an explicitly separate axisymmetric sensitivity analysis.
The target is a transparent worked analysis, not a new definitive Draco limit.

- Input: Walker et al. (2015) CDS table5, one weighted mean per unique star,
  1,565 rows. Raw SHA256:
  `ff889e669420da8e08a9d9e4495280fa8e45db9b64ebb938184589ac8f0e2c99`.
  Retain the table's quoted heliocentric velocities and uncertainties; do not
  treat repeated observations as independent stars.
- Explicit proposed sample: finite entries, `0 < e_vlos <= 5 km/s`,
  `-330 <= vlos <= -250 km/s`, `0 <= logg <= 3.5`,
  `-3.5 <= [Fe/H] <= -0.8`. These are **our numerical cuts**, not a claim to
  reproduce Walker's 468 by-eye members. Record every row's selection reason.
- Account for the velocity cut with a truncated Gaussian likelihood on the
  same window. Repeat with velocity bounds `[-340,-240]`, and separately with
  the stricter gravity/metallicity cuts `logg <= 3.0`, `[Fe/H] <= -1.0`.
  Foreground contamination and binaries remain limitations; include a
  repeated-observation subset comparison and posterior predictive residuals.
- Reference photometry: frozen 2012 McConnachie compilation, tables 1–3,
  including its original-source references. Center `(17h20m12.4s,+57d54m55s)`,
  distance `76 +/- 6 kpc`, semi-major half-light radius `10.00 +/- 0.30 arcmin`
  (`221 +/- 19 pc` including the distance uncertainty), PA `89 +/- 2 deg`
  east of north, ellipticity `0.31 +/- 0.02`. Use a tangent-plane projection
  with signed east/north coordinates followed by an explicit PA rotation.
- Baseline distance fixed at 76 kpc; separate 70/82 kpc sensitivity runs.
  The spherical tracer uses the circularized half-light scale, while the
  axisymmetric tracer uses the semi-major scale and `q_projected=0.69`.
  These are distinct geometric approximations, not an exactly matched model.
- Proposed spherical priors: `log10_rs_pc ~ Uniform(1.5,4.0)`,
  `log10_rhos_Msunpc3 ~ Uniform(-4,1)`,
  `log10(1-beta_ani) ~ Uniform(-0.5,1.0)`,
  `vmem_kms ~ Uniform(-310,-270)`. Photometric scale uses a Gaussian in log
  radius derived from the fixed-distance angular uncertainty. For J/D
  postprocessing adopt explicit 3/10 kpc finite halo-cutoff sensitivity cases;
  this cutoff is not Draco's empirical King stellar tidal radius.
- Do not run or interpret the real-data posterior until these model/prior
  choices and sensitivities have been accepted or replaced by the author.

Sources: [Walker et al.](https://academic.oup.com/mnras/article/448/3/2717/1091040),
[CDS catalogue](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/448/2717),
[McConnachie (2012)](https://arxiv.org/abs/1204.1562).
