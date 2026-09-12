# Current JAM 9 accuracy comparison

The first run of `jam9-plummer-isotropic-v1` used the protocol and runner
fixed in commit `51efc65`, before evaluating the new reference. Its raw
result, source hashes, failed coarse evaluations and follow-up warning
diagnostic are retained under `validation/release/`.

The two cases use a spherical Plummer gravitational potential with mass
$10^7\,M_\odot$ and scale 300 pc, a normalized Plummer tracer with intrinsic
$q=1$ or $0.7$, and inclination 1.1 rad. Both solvers use zero meridional
anisotropy. In this subset the spherical/cylindrical alignment distinction
vanishes; nonzero anisotropies do not define equivalent closures.

JAM receives the analytic potential. Its intrinsic calculation also accepts
an analytic tracer, whereas its projected solver requires a fitted MGE tracer.
The source and domain distinction follows [Cappellari (2026)](https://arxiv.org/abs/2601.16179)
and the [JamPy 9.0.2 documentation](https://pypi.org/project/jampy/9.0.2/).

| Intrinsic tracer q | Finest intrinsic difference | Finest projected difference | Projected JAM refinement | Finest MGE density error |
| --- | --- | --- | --- | --- |
| 1 | 2.3423e-10% | 0.0013537% | 0.056007% | 0.010275% |
| 0.7 | 1.21e-09% | 0.017715% | 0.33377% | 0.081937% |

Differences are maximum absolute relative differences over the six fixed
test positions, in the **second moment**, not its square root. Refinement
compares intermediate and fine configurations. The 0.5% cross-code and JAM
refinement criteria were fixed before execution. The MGE density error is
measured over 1–10,000 pc. It is an approximation diagnostic, not an error bar.

```{figure} ../_static/validation/jam9_validation.svg
:alt: Two Plummer tracer cases showing projected second moments and signed JAM residuals at three MGE, angular and LOS resolutions.
:width: 100%

Retained projected predictions at six positions. Lower panels show the
combined effect of tracer MGE fitting and projected-solver refinement;
these residuals do not isolate either contribution. Each marker is an actual
evaluation. There are no statistical uncertainty bars in this deterministic test.
```

## Coarse-grid failure and limits

JAM's intrinsic 45×9 radial/angular grid returned NaN at every tested
position for both tracer shapes. The 75×15 and 120×21 grids returned finite
results and satisfied the predefined refinement criterion. A bounded
follow-up located the warning in `jam_axi_intr.py`, line 253 of the hashed
9.0.2 source, where the solver takes the logarithm of $r\,\partial\Phi/\partial r$
for its outer Robin boundary condition. The spherical follow-up reproduced
the coarse failure and found no warning at either finer resolution or with
analytic spatial derivatives at the finest grid. The primary results were
not changed. A diagnostic JSON-writer failure on these NaNs and its repair
are recorded separately.

For the spherical case an independent Plummer formula gives

```{math}
\sigma_r^2(r)=\frac{GM}{6\sqrt{a^2+r^2}},\qquad
\overline{v_{\rm los}^2}(R)=\frac{3\pi GM}{64\sqrt{a^2+R^2}}.
```

The stored report includes comparisons against both analytic expressions.
The CI regression additionally compares the NumPy and JAX projected models
with the frozen JAM result and verifies the exact physical scaling
$\partial\overline{v_{\rm los}^2}/\partial\rho_s=\overline{v_{\rm los}^2}/\rho_s$.
It does not claim general finite-difference gradient validation from this one derivative.

This test covers no flattened gravitational potential, central cusp,
nonzero anisotropy, PSF/pixel averaging, foreground contamination or
posterior coverage. The recorded setup-inclusive times are diagnostics,
not a performance comparison or evidence of a faster solver.

## Reproduction

```bash
uv venv /tmp/jeanspy-jam9 --python 3.12
uv pip install --python /tmp/jeanspy-jam9/bin/python ".[plotting]" "jampy==9.0.2" "mgefit==6.2.6"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 600s prlimit --as=17179869184 \
  /tmp/jeanspy-jam9/bin/python scripts/validate_axisymmetric_jam9.py \
  --output /tmp/jam9-new-result.json
uv run --extra plotting python scripts/render_release_figures.py
```

The original environment versions are recorded in the JSON. A different
dependency environment is a new verification, not a byte-identical replay.
JAM's own license applies; its source is not included in the JeansPy distribution.
