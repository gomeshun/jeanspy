# Retained benchmark sources

The [matched LOS benchmark](los-benchmark.md) is a new experiment on the
release branch. The [forward timings](performance.md) and
[gradient checks](gradients.md) were completed earlier in a separate research
worktree. Their raw records, failures, numerical settings and original source
hashes are retained. Rendering these results does not rerun the experiments.

The three archives below contain the runtime and helper sources from the
pre-execution frozen commits. Each archive's 25 Python runtime files match
the hashes stored in its corresponding numerical report. Its protocol also
matches the recorded protocol hash. The two performance runners match their
recorded runner hashes. The gradient records did not record a runner hash;
that archive contains the runner from the stated frozen commit instead.
The {download}`archive manifest <../../../validation/release/retained_source_manifest.json>`
records every member hash and distinguishes these verification scopes.

| Study | Frozen commit | Source archive |
| --- | --- | --- |
| Physical gradients, including failed gates | `1939ab7` | {download}`Gradient source <../../../validation/release/retained-gradients-source.tar.gz>` |
| Original forward benchmark, including 16 failed spherical setup attempts | `704b762` | {download}`Primary forward source <../../../validation/release/retained-performance-primary-source.tar.gz>` |
| Corrected spherical forward benchmark | `c74aeb7` | {download}`Corrected spherical source <../../../validation/release/retained-performance-spherical-v2-source.tar.gz>` |

The correction supplied the missing classical NFW cutoff as infinity. It did
not change the physical model, runtime numerical code, repetition counts or
accuracy thresholds. The original exceptions remain in the primary records.

The retained worktree and the current release runtime have also been compared
as Python syntax trees with only leading docstrings removed. All 25 files are
equivalent under that comparison; three files differ in documentation text.
This establishes code equivalence for that comparison, not fresh timing or
scientific calibration. The
{download}`comparison record <../../../validation/release/retained_runtime_comparison.json>`
identifies both commits and every file hash. The frozen archives remain the
source of truth for exact historical source identity.

## Rebuild the displayed results

These commands read the committed records and produce the pages, figures and
machine-readable tables without numerical experiments:

```bash
python scripts/render_release_performance.py
python scripts/render_release_gradients.py
```

Their figure manifests retain hashes of every input and output. All raw
predictions, derivatives, warnings, repeated durations and supervisor records
are in `validation/release/campaign/`. Precision and dependency versions are
recorded per condition. No posterior inference outputs are needed by these
renderers.

## A new numerical execution

Extract the appropriate archive into a fresh directory. It contains `src/`,
`scripts/`, the package configuration and its frozen protocol. Use the Python
and dependency versions recorded by the corresponding original condition.
The performance entry points are `scripts/benchmark_release_performance.py`
and `scripts/benchmark_release_spherical_performance.py`; the gradient entry
point is `scripts/validate_physical_gradients.py`. Each takes `--output` with
a fresh path. The archived primary performance runner intentionally retains
the original setup defect; use the corrected archive for spherical timings.

Set `JEANSPY_JAX_PLATFORM=cpu` or `gpu`, `JEANSPY_JAX_ENABLE_X64=true` or
`false`, and `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` before starting Python.
The recorded CPU affinity is `taskset -c 0-7`. Run through the release tree's
`scripts/run_budgeted_experiment.py` using the selected protocol's declared
time bound, the existing canonical campaign ledger and a new experiment
label. Its supervisor enforces host/device memory bounds and preserves
partial output. A new output directory does not reset the cumulative budget.
The author's release decision excludes resuming the earlier MCMC queue.
