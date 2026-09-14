# Synchronized forward-model performance

These measurements evaluate fixed physical points and synthetic coordinates; they do not measure posterior convergence. Each JAX result is synchronized before the timer stops. Shading shows the 10th–90th percentiles of 20 repeated evaluations, not a confidence interval over independent hardware sessions. The NumPy reference uses five repetitions. Spherical and axisymmetric panels use different vertical scales; compare their labeled values.

The primary protocol was fixed in commit `704b762`. Its 16 spherical attempts failed because the harness omitted the required classical NFW truncation parameter. Those exceptions remain in the primary records. The corrected protocol and runner, fixed in commit `c74aeb7` after diagnosing this setup error, explicitly set the same infinite cutoff for both backends and repeat only the spherical measurements. No runtime library code, physical value, timing repetition or acceptance threshold was changed.

```{figure} ../_static/validation/forward_performance.svg
:alt: Warm prediction and four-parameter likelihood-gradient timings versus data count, for four device and precision conditions.
:width: 100%

Prediction and scalar Gaussian log-likelihood gradient timings for spherical NFW and axisymmetric Zhao models. The NumPy spherical double-exponential and JAX transformed quadratures solve the same physical problem with different numerical rules. Axisymmetric backends use the same 32-by-32-by-32 rule. The CPU is an AMD Threadripper 3990X restricted to logical CPUs 0–7; the GPU is one RTX 3090. BLAS/OpenMP thread counts are one.
```

All 32 completed geometry/data-count/condition cases pass their declared checks: finite positive values, at most 0.5% NumPy/JAX prediction differences, finite derivatives, and, for axisymmetry, at most 0.5% order-32/order-64 prediction differences at up to 16 fixed positions. These checks do not replace the broader [physical-gradient study](gradients.md), which finds nonfinite float32 axisymmetric gradients in separate tests with different physical points and quadrature orders.

## Measured stages at 256 positions

All entries below are milliseconds. First-call time includes tracing, compilation and execution. Host input transfer and result transfer are measured separately; CPU device-put is also reported. Warm values are medians. The gradient has four active physical parameters.

| Geometry | Condition | Host preparation | Input transfer | First prediction | Warm prediction | First value + gradient | Warm value + gradient | Prediction to host |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spherical | cpu-float64 | 0.2004 | 0.4681 | 268.7 | 1.449 | 360.9 | 1.967 | 0.02528 |
| axisymmetric | cpu-float64 | 0.1685 | 0.432 | 732.9 | 353 | 2613 | 1118 | 0.01967 |
| spherical | gpu-float64 | 0.2244 | 0.6959 | 450.3 | 0.1461 | 955.2 | 0.4972 | 0.1579 |
| axisymmetric | gpu-float64 | 0.1657 | 0.7287 | 888.9 | 31.65 | 4592 | 184.6 | 0.1676 |
| spherical | cpu-float32 | 0.1952 | 0.4784 | 250.8 | 0.3068 | 368.7 | 1.376 | 0.02098 |
| axisymmetric | cpu-float32 | 0.1779 | 0.449 | 553.4 | 108.2 | 2290 | 626.6 | 0.02508 |
| spherical | gpu-float32 | 0.1813 | 0.7274 | 371 | 0.08026 | 823.3 | 0.1207 | 0.1704 |
| axisymmetric | gpu-float32 | 0.1657 | 0.787 | 728.7 | 4.452 | 3565 | 30.97 | 0.21 |

```{figure} ../_static/validation/gradient_parameter_performance.svg
:alt: Value and gradient timings versus number of active parameters at 256 positions.
:width: 100%

Changing the differentiated subset leaves the physical point and data fixed. Nearly constant axisymmetric costs across 1–10 parameters reflect this reverse-mode workload and are not a general scaling guarantee for additional model components.
```

## Complete warm timing table

All times are milliseconds. NumPy timings are float64 in every condition; its repeated measurements alongside different JAX conditions are retained as measured.

| Geometry | Condition | Positions | NumPy prediction | JAX prediction | JAX value + gradient (4 parameters) | Max NumPy/JAX relative error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| spherical | cpu-float64 | 8 | 0.5382 | 0.1401 | 0.333 | 2.27e-14 |
| spherical | cpu-float64 | 64 | 3.221 | 0.4395 | 1.054 | 2.35e-14 |
| spherical | cpu-float64 | 256 | 17.81 | 1.449 | 1.967 | 2.3e-14 |
| spherical | cpu-float64 | 1024 | 52.09 | 2.284 | 4.662 | 2.33e-14 |
| axisymmetric | cpu-float64 | 8 | 49.62 | 10.24 | 35.46 | 5.35e-16 |
| axisymmetric | cpu-float64 | 64 | 392.8 | 81.53 | 285 | 5.13e-16 |
| axisymmetric | cpu-float64 | 256 | 1600 | 353 | 1118 | 1.03e-15 |
| axisymmetric | cpu-float64 | 1024 | 6377 | 1385 | 4422 | 1.01e-15 |
| spherical | gpu-float64 | 8 | 0.5355 | 0.1467 | 0.235 | 2.27e-14 |
| spherical | gpu-float64 | 64 | 2.275 | 0.1396 | 0.3422 | 2.3e-14 |
| spherical | gpu-float64 | 256 | 16.25 | 0.1461 | 0.4972 | 2.29e-14 |
| spherical | gpu-float64 | 1024 | 51.47 | 0.233 | 0.9761 | 2.3e-14 |
| axisymmetric | gpu-float64 | 8 | 49.75 | 1.169 | 6.646 | 1.78e-16 |
| axisymmetric | gpu-float64 | 64 | 394.5 | 9.443 | 46.69 | 3.53e-16 |
| axisymmetric | gpu-float64 | 256 | 1584 | 31.65 | 184.6 | 5.66e-16 |
| axisymmetric | gpu-float64 | 1024 | 6337 | 127 | 742.9 | 6.57e-16 |
| spherical | cpu-float32 | 8 | 0.5443 | 0.08558 | 0.2133 | 2.39e-07 |
| spherical | cpu-float32 | 64 | 3.223 | 0.1766 | 0.6342 | 4.57e-07 |
| spherical | cpu-float32 | 256 | 17.75 | 0.3068 | 1.376 | 4.65e-07 |
| spherical | cpu-float32 | 1024 | 52.03 | 0.7064 | 2.952 | 5.42e-07 |
| axisymmetric | cpu-float32 | 8 | 50.02 | 3.262 | 21.22 | 1.67e-07 |
| axisymmetric | cpu-float32 | 64 | 399.1 | 26.41 | 157.6 | 4.04e-07 |
| axisymmetric | cpu-float32 | 256 | 1610 | 108.2 | 626.6 | 3.81e-07 |
| axisymmetric | cpu-float32 | 1024 | 6459 | 414.9 | 2499 | 5.45e-07 |
| spherical | gpu-float32 | 8 | 0.542 | 0.07446 | 0.1119 | 5.68e-07 |
| spherical | gpu-float32 | 64 | 2.286 | 0.07976 | 0.1141 | 7.74e-07 |
| spherical | gpu-float32 | 256 | 17.17 | 0.08026 | 0.1207 | 8.57e-07 |
| spherical | gpu-float32 | 1024 | 51.32 | 0.08593 | 0.2717 | 9.65e-07 |
| axisymmetric | gpu-float32 | 8 | 50.01 | 0.2338 | 1.309 | 4.92e-07 |
| axisymmetric | gpu-float32 | 64 | 396.8 | 1.198 | 8.546 | 3.33e-07 |
| axisymmetric | gpu-float32 | 256 | 1581 | 4.452 | 30.97 | 5.57e-07 |
| axisymmetric | gpu-float32 | 1024 | 6360 | 16.32 | 122.3 | 5.9e-07 |

## Memory scope and reproducibility

These are supervisor-observed process high-water marks across all four sizes and all parameter subsets in one condition. GPU allocation includes allocator/runtime reservations. They are not isolated incremental kernel requirements. The primary axisymmetric condition also includes the retained failed spherical setup attempts.

| Geometry | Condition | Peak host RSS [GiB] | Observed GPU allocation [GiB] |
| --- | --- | ---: | ---: |
| spherical | cpu-float64 | 0.414 | 0.000 |
| axisymmetric | cpu-float64 | 0.629 | 0.000 |
| spherical | gpu-float64 | 0.906 | 0.264 |
| axisymmetric | gpu-float64 | 0.970 | 0.273 |
| spherical | cpu-float32 | 0.405 | 0.000 |
| axisymmetric | cpu-float32 | 0.646 | 0.000 |
| spherical | gpu-float32 | 0.895 | 0.264 |
| axisymmetric | gpu-float32 | 0.991 | 0.268 |

Every repetition, warning, first call, transfer, model-construction cost, derivative and source hash is retained in `validation/release/campaign/performance-*.json` and `spherical-performance-*-v2.json`; supervisor records give process limits and memory observations. `validation/release/performance_summary.json` ties the displayed rows to those records. Regenerate the public figures and this page with `python scripts/render_release_performance.py`. The source archives and reproduction commands on the [retained benchmark source page](retained-sources.md) preserve the original runner and runtime hashes. Current runtime code is equivalent after removing docstrings, as separately checked; these timings were not rerun on the release branch. New experiments must respect the existing cumulative-budget boundary.
