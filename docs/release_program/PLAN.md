# JeansPy documentation, comparative validation and technical paper

<!-- site-start -->

Baseline: `1a0ad4028d26af1df389ebdfdf992285ec50f8bb`. The public runtime API and
scientific definitions are preserved. Work started 2026-09-12; the baseline
main Test workflow passed (run 34696825360). Pages was not yet configured,
the paper submodule contained only a README, and no PR was open at inspection.

| Milestone | State | Completion evidence required |
| --- | --- | --- |
| English documentation | In progress | Public URL; complete API contracts; executable examples; warning-free build; browser QA; three full analyses |
| Competitors and venue | In progress | Primary-source matrix, bibliography, claims/tasks; author selects venue |
| Reproducible benchmarks | Selected accuracy validation complete; broader protocol pending | Frozen plan, environments, seeds, raw successes/failures, regenerable tables/figures |
| Submission package | Not started | Evidence-linked manuscript in `jeanspy_paper`, selected format, references, supplements, availability statements, author review |

Completed foundation: 74 public API exports and method contracts, nine executable
examples, a warning-free HTML/doctest build, a local HTML file/anchor audit,
version-preservation checks, a 15-row source-backed software matrix and 34
bibliographic records. The two frozen JAM 9 isotropic comparisons retain both
the passing fine grids and the failed coarse grid. The full repository suite
passed 469 tests and 37 subtests in the locked CPU environment.

The proposed Draco scientific recipe, production compute envelope and venue
ranking are in `docs/release_program/AUTHOR_DECISIONS.md`; author choices remain
pending. Full production inference, repeated-mock coverage and the three
completed analysis tutorials have not yet been delivered. Experiment conditions
must be frozen before results are inspected. Pilot runs cannot be relabeled as
preregistered evidence.

CPU float64 is the reference. CPU float32 and actual GPU measurements are
separate conditions. The local host has an AMD Threadripper 3990X and an
NVIDIA RTX 3090, confirmed by an unsandboxed read-only hardware inspection.
The complete environment and actual device must still be recorded per run.

The central claim concerns the usefulness of physical-parameter gradients in
direct spherical/axisymmetric inference at controlled accuracy. NumPy J/D
postprocessing is outside that differentiability claim. Unfavorable results,
nonconvergence, installation failures and incomparable tasks remain reportable
outcomes. Sampling speed and ESS are not scientific-calibration tests.

The target is submission readiness. Creating a journal submission, handling
referees and achieving acceptance are outside this program.
