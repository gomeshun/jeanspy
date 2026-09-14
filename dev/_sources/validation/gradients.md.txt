# Physical-parameter gradient checks

The protocol was fixed in commit `1939ab7` before the four CPU/GPU and precision conditions. Each runs all seven selected cases; all conditions completed but retain failed numerical gates. This is a selected-point numerical study, not a guarantee over a prior or a calibration test.

```{figure} ../_static/validation/physical_gradients.svg
:alt: Four device and precision conditions show largest derivative errors and refinement errors for seven models, including failures for finite truncation.
:width: 100%

The top panel takes the larger error from the two fixed finite-difference steps at the finest quadrature. The bottom panel compares intermediate and fine quadratures. Values are maxima over positions and physical parameters; they are deterministic discrepancies without statistical error bars. Full per-parameter values are in `validation/release/gradient_parameter_summary.json`.
```

For each parameter $p_j$, the scale is $s_j=\max(|p_j|,0.1)$ in its native units. The displayed discrepancy is $\max_i |\Delta(\partial f_i/\partial p_j)|s_j/|f_i|$. It remains well-defined when the derivative itself is near zero. The 0.005 derivative-refinement gate is distinct from same-order AD/finite-difference agreement.

| Condition | Case | Finest AD/FD max | Parameter | Gradient refinement max | Parameter | All declared gates |
| --- | --- | ---: | --- | ---: | --- | --- |
| cpu-float64 | nfw-constant | 1.202e-09 | `r_t_pc` | 0.005645 | `r_t_pc` | fail |
| cpu-float64 | nfw-om | 2.532e-08 | `r_t_pc` | 0.01208 | `r_t_pc` | fail |
| cpu-float64 | nfw-baes | 1.652e-12 | `rs_pc` | 0.003097 | `r_t_pc` | pass |
| cpu-float64 | zhao-constant | 0.0008483 | `r_t_pc` | 0.01359 | `r_t_pc` | fail |
| cpu-float64 | oblate-cusp | 2.652e-12 | `beta` | 2.218e-08 | `gamma` | pass |
| cpu-float64 | prolate-halo-core | 1.765e-12 | `alpha` | 2.395e-10 | `rhos_Msunpc3` | pass |
| cpu-float64 | finite-halo | 0.3077 | `r_t_pc` | 0.02196 | `r_t_pc` | fail |
| gpu-float64 | nfw-constant | 1.202e-09 | `r_t_pc` | 0.005645 | `r_t_pc` | fail |
| gpu-float64 | nfw-om | 2.532e-08 | `r_t_pc` | 0.01208 | `r_t_pc` | fail |
| gpu-float64 | nfw-baes | 1.059e-12 | `re_pc` | 0.003097 | `r_t_pc` | pass |
| gpu-float64 | zhao-constant | 0.0008483 | `r_t_pc` | 0.01359 | `r_t_pc` | fail |
| gpu-float64 | oblate-cusp | 2.668e-12 | `beta` | 2.218e-08 | `gamma` | pass |
| gpu-float64 | prolate-halo-core | 1.605e-12 | `alpha` | 2.395e-10 | `rhos_Msunpc3` | pass |
| gpu-float64 | finite-halo | 0.3077 | `r_t_pc` | 0.02196 | `r_t_pc` | fail |
| cpu-float32 | nfw-constant | 0.001116 | `re_pc` | 0.005645 | `r_t_pc` | fail |
| cpu-float32 | nfw-om | 0.001244 | `re_pc` | 0.01208 | `r_t_pc` | fail |
| cpu-float32 | nfw-baes | 0.0006694 | `re_pc` | 0.003097 | `r_t_pc` | pass |
| cpu-float32 | zhao-constant | 0.001191 | `re_pc` | 0.01359 | `r_t_pc` | fail |
| cpu-float32 | oblate-cusp | nonfinite | `re_pc` | nonfinite | `re_pc` | fail |
| cpu-float32 | prolate-halo-core | nonfinite | `re_pc` | nonfinite | `re_pc` | fail |
| cpu-float32 | finite-halo | nonfinite | `re_pc` | nonfinite | `re_pc` | fail |
| gpu-float32 | nfw-constant | 0.002041 | `rs_pc` | 0.005645 | `r_t_pc` | fail |
| gpu-float32 | nfw-om | 0.0006749 | `rs_pc` | 0.01208 | `r_t_pc` | fail |
| gpu-float32 | nfw-baes | 0.004973 | `rs_pc` | 0.003097 | `r_t_pc` | pass |
| gpu-float32 | zhao-constant | 0.0009393 | `r_t_pc` | 0.01359 | `r_t_pc` | fail |
| gpu-float32 | oblate-cusp | nonfinite | `re_pc` | nonfinite | `re_pc` | fail |
| gpu-float32 | prolate-halo-core | nonfinite | `re_pc` | nonfinite | `re_pc` | fail |
| gpu-float32 | finite-halo | nonfinite | `re_pc` | nonfinite | `re_pc` | fail |

The final column includes both finite-difference steps, every evaluated order, reference values and refinement. It can fail even when the two displayed finest-order summaries look acceptable. The smooth untruncated axisymmetric cases pass in CPU/GPU float64. All three axisymmetric float32 cases have nonfinite derivatives in both device conditions; the x markers at ordinate 1 denote these failures, not numerical error values. For a nonfinite table entry the parameter names the first nonfinite column, and the machine-readable summary stores null rather than a numerical discrepancy. Finite-cutoff cases need a local convergence study before using the affected derivatives for physical claims. No threshold, seed or resolution was changed in response to these outputs.

The spherical CPU float64 value checks all pass the 0.5% reference threshold. Finite hard-cutoff derivative discrepancies can be much larger despite accurate values. A small-step finite difference alone can conceal step instability, as the Zhao truncation-radius example demonstrates. The observations are consistent with moving-cutoff sensitivity but do not isolate a universal mechanism.

Regenerate the figure and table with `python scripts/render_release_gradients.py`. Raw predictions, Jacobians, both finite differences, source hashes, environments and warnings are in `validation/release/campaign/gradients-*.json`. These records include diagnostic timing fields, which are not dedicated performance measurements.
