# JeansPy release preparation

<!-- site-start -->

Baseline: `1a0ad4028d26af1df389ebdfdf992285ec50f8bb`. The public runtime API and
scientific definitions are preserved. On 2026-09-14 (JST), the author
prioritized the software release and deferred the paper and further
core/cusp-prior research. The release programme covers published English
documentation, reproducible benchmarks and preparation for a PyPI release.

| Deliverable | Current evidence | Remaining release work |
| --- | --- | --- |
| English documentation | API docstrings, runnable examples, theory, numerical contracts and versioned-site workflow | Review development and release builds; deploy and verify the public site |
| Reproducible benchmarks | New frozen LOS accuracy/time experiment: 99 settings, 98 passing the 0.5% criterion; 32 retained forward timing cases and the original 16 setup failures; physical-gradient failures preserved | Review the English figures/tables and exact source archives; no further MCMC comparison |
| PyPI release preparation | Test matrix, installed-artifact checks and source-content validation | Review the exact candidate, validate release artifacts and verify publisher setup; actual upload requires separate authorization |

The existing fifteen-entry software comparison and bibliography are retained
as background. Further software/venue research and a submission-ready paper
are outside the active objective.

API descriptions live beside the implementation. Short examples demonstrate
construction, numerical evaluation, inference and persisted restart. Their
small sample counts do not establish convergence or statistical calibration.
The longer synthetic and real-data studies remain research work; a completed
manuscript is not a software release requirement.

Numerical tests, sampler convergence and scientific calibration remain distinct
claims. Benchmark tables must preserve input provenance, versions, precision,
devices, seeds, numerical settings and predeclared stopping rules. Retain failed
or incomplete runs and identify incomparable models or likelihoods. A numerical
accuracy result from selected cases does not establish the entire prior domain,
and sampling speed or ESS does not establish calibrated uncertainty.

The separate research campaign and manuscript are retained without restarting
additional experiments. Review existing evidence for the release before
bringing it into this branch. Do not silently substitute new seeds, relax
diagnostics or relabel an author-directed amendment as a preregistered run.

On 2026-09-14 the author excluded MCMC comparisons because of their cost and
prioritized line-of-sight dispersion accuracy and evaluation time. The new
`matched-los-accuracy-time-v1` protocol fixes matched spherical/oblate Plummer
cases, three position counts, three numerical resolutions and ten prescribed
mass-normalization changes. It compares the public NumPy/JAX solvers and
JAM 9.0.2 against an independent reference, separating MGE fitting, input
transfer, first-call compilation, warm evaluation and output transfer. Its
runtime is charged to the existing campaign budget. The familiar physical
cases come from the earlier JAM validation; only the new experiment is frozen
before its first execution. Failed settings remain part of the report.

Main publishes development documentation. A published GitHub Release adds an
immutable documentation version; the highest formal release supplies `stable`.
The package tag and GitHub Release are separate required steps in the
[release procedure](https://github.com/gomeshun/jeanspy/blob/main/RELEASE.md).

The author must review the planned public result and explicitly approve
merging. GitHub Pages deployment and package publication remain pending that
review. Historical author choices are retained in
`docs/release_program/AUTHOR_DECISIONS.md`.
