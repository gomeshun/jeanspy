# LOS dispersion: accuracy and time

This forward-only benchmark compares JeansPy's NumPy and JAX solvers with JAM 9.0.2 on matched Plummer models. MCMC comparison is outside the release scope by the author's 2026-09-14 decision. The protocol and runner were fixed in commit `b3f1545` before any new evaluation. The physical cases and JAM settings were already studied in the earlier [JAM validation](jam9.md); this is a fresh timing experiment, not an unseen-model test.

## Physical comparison

The potential is spherical Plummer with mass $10^7 M_\odot$ and scale 300 pc. The tracer has the same scale, intrinsic axis ratio $q=1$ or $0.7$, inclination 1.1 rad, and zero meridional anisotropy. Coordinates follow the same projected ellipses from 10 to 1500 pc. The observable is $\overline{v_{\rm los}^2}$ in $(\mathrm{km\,s^{-1}})^2$. Its square root is $\sigma_{\rm los}$ for the imposed zero-mean-velocity convention, and $v_{\rm rms}$ otherwise. There is no PSF, pixel averaging, measurement error or fitted streaming model.

JAM's spherical alignment and JeansPy's cylindrical alignment give the same even-moment equations in this zero-anisotropy subset. JAM takes an analytic potential containing the same gravitational constant, plus a fitted MGE tracer. The MGE approximation is part of its reported total prediction error. Its fitting time is reported separately and is reusable while the tracer is fixed. The specialized spherical JeansPy solver is included only for $q=1$. The general axisymmetric solver remains present in that case for a like-geometry comparison. See [Cappellari (2026)](https://arxiv.org/abs/2601.16179) and the [pinned JAM documentation](https://pypi.org/project/jampy/9.0.2/).

## Accuracy versus time

```{figure} ../_static/validation/los_accuracy_time.svg
:alt: Six panels compare LOS second-moment error against synchronized warm time for two tracer shapes, three position counts, three numerical settings and four CPU or GPU implementations.
:width: 100%

Each marker is a measured setting; connecting segments only identify the three fixed configurations. Error is the largest relative discrepancy over all positions and all eleven mass normalizations. Horizontal bars show the 10th–90th percentiles of ten warm calls, not uncertainty across independent machine sessions. GPU timings synchronize completion and exclude separately recorded transfers. The dotted line is the predeclared 0.5% second-moment criterion. Errors below $10^{-14}$ are plotted at that floor, with exact values retained in the tables and raw data.
```

All 99 declared rows were attempted; 98 pass the 0.5% criterion and 1 do not. A completed timing row can fail numerical accuracy. The full table below retains every setting; no fastest-setting selection or universal speedup is inferred.

All first-call times include the ordinary forward evaluation; JAX also includes tracing and compilation. Warm calls use the same ten prescribed mass-normalization changes (within one percent), with input shapes and tracer fixed. JAX physical parameters and coordinates are dynamic arguments. NumPy and JAM recompute their public forward call. No output is substituted by rescaling an earlier prediction.

## Complete measured settings

Times are milliseconds. The maximum error includes the first call and every warm prediction. The square-root error is computed directly from the same moment ratio. JAM settings list MGE component count, angular order and LOS order; JeansPy settings give its fixed quadrature order. Spherical mass quadrature has 128 nodes in both JeansPy implementations.

| Engine | Solver | q | Positions | Setting | First [ms] | Warm median [ms] | Moment error | Sigma error | 0.5% gate |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| JeansPy NumPy | axisymmetric | 1 | 16 | 16 | 19.97 | 19.46 | 7.745e-07 | 3.872e-07 | pass |
| JeansPy NumPy | axisymmetric | 1 | 16 | 32 | 128 | 127.8 | 8.333e-10 | 4.166e-10 | pass |
| JeansPy NumPy | axisymmetric | 1 | 16 | 64 | 878.8 | 878.5 | 8.689e-13 | 4.343e-13 | pass |
| JeansPy NumPy | spherical | 1 | 16 | 128 | 18.85 | 16.27 | 3.245e-08 | 1.623e-08 | pass |
| JeansPy NumPy | spherical | 1 | 16 | 256 | 28.79 | 28.25 | 9.77e-15 | 4.885e-15 | pass |
| JeansPy NumPy | spherical | 1 | 16 | 512 | 54.36 | 51.38 | 4.663e-15 | 2.22e-15 | pass |
| JeansPy NumPy | axisymmetric | 1 | 64 | 16 | 77.74 | 77.65 | 7.741e-07 | 3.871e-07 | pass |
| JeansPy NumPy | axisymmetric | 1 | 64 | 32 | 356.8 | 354.4 | 8.332e-10 | 4.166e-10 | pass |
| JeansPy NumPy | axisymmetric | 1 | 64 | 64 | 3519 | 3512 | 8.646e-13 | 4.323e-13 | pass |
| JeansPy NumPy | spherical | 1 | 64 | 128 | 51.89 | 51.68 | 9.968e-08 | 4.984e-08 | pass |
| JeansPy NumPy | spherical | 1 | 64 | 256 | 104.1 | 100.4 | 9.77e-15 | 4.885e-15 | pass |
| JeansPy NumPy | spherical | 1 | 64 | 512 | 203.5 | 203.5 | 4.885e-15 | 2.442e-15 | pass |
| JeansPy NumPy | axisymmetric | 1 | 256 | 16 | 310.2 | 309.9 | 7.744e-07 | 3.872e-07 | pass |
| JeansPy NumPy | axisymmetric | 1 | 256 | 32 | 1414 | 1411 | 8.332e-10 | 4.166e-10 | pass |
| JeansPy NumPy | axisymmetric | 1 | 256 | 64 | 1.389e+04 | 1.387e+04 | 8.622e-13 | 4.31e-13 | pass |
| JeansPy NumPy | spherical | 1 | 256 | 128 | 204.1 | 204.3 | 9.984e-08 | 4.992e-08 | pass |
| JeansPy NumPy | spherical | 1 | 256 | 256 | 395.1 | 394.4 | 9.992e-15 | 4.885e-15 | pass |
| JeansPy NumPy | spherical | 1 | 256 | 512 | 780.5 | 779.1 | 4.885e-15 | 2.442e-15 | pass |
| JeansPy NumPy | axisymmetric | 0.7 | 16 | 16 | 19.76 | 19.62 | 2.188e-07 | 1.094e-07 | pass |
| JeansPy NumPy | axisymmetric | 0.7 | 16 | 32 | 88.61 | 88.62 | 2.375e-10 | 1.188e-10 | pass |
| JeansPy NumPy | axisymmetric | 0.7 | 16 | 64 | 863.4 | 862.7 | 2.476e-13 | 1.237e-13 | pass |
| JeansPy NumPy | axisymmetric | 0.7 | 64 | 16 | 77.8 | 77.85 | 1.185e-07 | 5.923e-08 | pass |
| JeansPy NumPy | axisymmetric | 0.7 | 64 | 32 | 354.1 | 353 | 1.297e-10 | 6.485e-11 | pass |
| JeansPy NumPy | axisymmetric | 0.7 | 64 | 64 | 3462 | 3461 | 1.321e-13 | 6.595e-14 | pass |
| JeansPy NumPy | axisymmetric | 0.7 | 256 | 16 | 310 | 310.1 | 1.723e-07 | 8.616e-08 | pass |
| JeansPy NumPy | axisymmetric | 0.7 | 256 | 32 | 1416 | 1415 | 1.846e-10 | 9.232e-11 | pass |
| JeansPy NumPy | axisymmetric | 0.7 | 256 | 64 | 1.386e+04 | 1.385e+04 | 1.921e-13 | 9.592e-14 | pass |
| JeansPy JAX CPU | axisymmetric | 1 | 16 | 16 | 529.3 | 6.358 | 7.745e-07 | 3.872e-07 | pass |
| JeansPy JAX CPU | axisymmetric | 1 | 16 | 32 | 516.2 | 13.41 | 8.333e-10 | 4.166e-10 | pass |
| JeansPy JAX CPU | axisymmetric | 1 | 16 | 64 | 614.1 | 144.9 | 8.686e-13 | 4.343e-13 | pass |
| JeansPy JAX CPU | spherical | 1 | 16 | 32 | 260.6 | 0.6843 | 0.0007721 | 0.000386 | pass |
| JeansPy JAX CPU | spherical | 1 | 16 | 64 | 266.4 | 1.31 | 1.44e-08 | 7.198e-09 | pass |
| JeansPy JAX CPU | spherical | 1 | 16 | 128 | 295.7 | 2.313 | 8.882e-15 | 4.441e-15 | pass |
| JeansPy JAX CPU | axisymmetric | 1 | 64 | 16 | 458.7 | 15.43 | 7.741e-07 | 3.871e-07 | pass |
| JeansPy JAX CPU | axisymmetric | 1 | 64 | 32 | 605.1 | 53.69 | 8.332e-10 | 4.166e-10 | pass |
| JeansPy JAX CPU | axisymmetric | 1 | 64 | 64 | 1051 | 585.3 | 8.646e-13 | 4.323e-13 | pass |
| JeansPy JAX CPU | spherical | 1 | 64 | 32 | 285.6 | 2.326 | 0.0008271 | 0.0004136 | pass |
| JeansPy JAX CPU | spherical | 1 | 64 | 64 | 274.6 | 4.337 | 3.444e-08 | 1.722e-08 | pass |
| JeansPy JAX CPU | spherical | 1 | 64 | 128 | 299.6 | 8.495 | 8.882e-15 | 4.441e-15 | pass |
| JeansPy JAX CPU | axisymmetric | 1 | 256 | 16 | 540.8 | 61.06 | 7.744e-07 | 3.872e-07 | pass |
| JeansPy JAX CPU | axisymmetric | 1 | 256 | 32 | 751.2 | 235.6 | 8.332e-10 | 4.166e-10 | pass |
| JeansPy JAX CPU | axisymmetric | 1 | 256 | 64 | 2851 | 2336 | 8.622e-13 | 4.31e-13 | pass |
| JeansPy JAX CPU | spherical | 1 | 256 | 32 | 286.1 | 9.268 | 0.0008562 | 0.0004282 | pass |
| JeansPy JAX CPU | spherical | 1 | 256 | 64 | 298.2 | 22.03 | 3.439e-08 | 1.72e-08 | pass |
| JeansPy JAX CPU | spherical | 1 | 256 | 128 | 382.9 | 41.19 | 8.882e-15 | 4.441e-15 | pass |
| JeansPy JAX CPU | axisymmetric | 0.7 | 16 | 16 | 464.2 | 7.872 | 2.188e-07 | 1.094e-07 | pass |
| JeansPy JAX CPU | axisymmetric | 0.7 | 16 | 32 | 513.6 | 24.22 | 2.375e-10 | 1.188e-10 | pass |
| JeansPy JAX CPU | axisymmetric | 0.7 | 16 | 64 | 606.4 | 148.2 | 2.476e-13 | 1.237e-13 | pass |
| JeansPy JAX CPU | axisymmetric | 0.7 | 64 | 16 | 488.2 | 30.8 | 1.185e-07 | 5.923e-08 | pass |
| JeansPy JAX CPU | axisymmetric | 0.7 | 64 | 32 | 590.7 | 86.09 | 1.297e-10 | 6.485e-11 | pass |
| JeansPy JAX CPU | axisymmetric | 0.7 | 64 | 64 | 1059 | 583.8 | 1.323e-13 | 6.617e-14 | pass |
| JeansPy JAX CPU | axisymmetric | 0.7 | 256 | 16 | 560.4 | 61.19 | 1.723e-07 | 8.616e-08 | pass |
| JeansPy JAX CPU | axisymmetric | 0.7 | 256 | 32 | 770.8 | 246.2 | 1.846e-10 | 9.232e-11 | pass |
| JeansPy JAX CPU | axisymmetric | 0.7 | 256 | 64 | 2792 | 2337 | 1.923e-13 | 9.615e-14 | pass |
| JeansPy JAX GPU | axisymmetric | 1 | 16 | 16 | 1141 | 1.578 | 7.745e-07 | 3.872e-07 | pass |
| JeansPy JAX GPU | axisymmetric | 1 | 16 | 32 | 1019 | 2.588 | 8.333e-10 | 4.166e-10 | pass |
| JeansPy JAX GPU | axisymmetric | 1 | 16 | 64 | 1257 | 10.19 | 8.689e-13 | 4.343e-13 | pass |
| JeansPy JAX GPU | spherical | 1 | 16 | 32 | 540.7 | 0.2223 | 0.0007721 | 0.000386 | pass |
| JeansPy JAX GPU | spherical | 1 | 16 | 64 | 1046 | 0.3351 | 1.44e-08 | 7.198e-09 | pass |
| JeansPy JAX GPU | spherical | 1 | 16 | 128 | 1097 | 0.5815 | 8.882e-15 | 4.441e-15 | pass |
| JeansPy JAX GPU | axisymmetric | 1 | 64 | 16 | 1079 | 5.908 | 7.741e-07 | 3.871e-07 | pass |
| JeansPy JAX GPU | axisymmetric | 1 | 64 | 32 | 917.6 | 9.819 | 8.332e-10 | 4.166e-10 | pass |
| JeansPy JAX GPU | axisymmetric | 1 | 64 | 64 | 1285 | 36.44 | 8.649e-13 | 4.323e-13 | pass |
| JeansPy JAX GPU | spherical | 1 | 64 | 32 | 1008 | 0.4406 | 0.0008271 | 0.0004136 | pass |
| JeansPy JAX GPU | spherical | 1 | 64 | 64 | 1088 | 0.6524 | 3.444e-08 | 1.722e-08 | pass |
| JeansPy JAX GPU | spherical | 1 | 64 | 128 | 1791 | 0.9893 | 8.882e-15 | 4.441e-15 | pass |
| JeansPy JAX GPU | axisymmetric | 1 | 256 | 16 | 1098 | 21.06 | 7.744e-07 | 3.872e-07 | pass |
| JeansPy JAX GPU | axisymmetric | 1 | 256 | 32 | 949.5 | 33.25 | 8.332e-10 | 4.166e-10 | pass |
| JeansPy JAX GPU | axisymmetric | 1 | 256 | 64 | 1428 | 143.4 | 8.622e-13 | 4.31e-13 | pass |
| JeansPy JAX GPU | spherical | 1 | 256 | 32 | 1726 | 1.001 | 0.0008562 | 0.0004282 | pass |
| JeansPy JAX GPU | spherical | 1 | 256 | 64 | 1927 | 1.755 | 3.439e-08 | 1.72e-08 | pass |
| JeansPy JAX GPU | spherical | 1 | 256 | 128 | 5086 | 3.004 | 8.882e-15 | 4.441e-15 | pass |
| JeansPy JAX GPU | axisymmetric | 0.7 | 16 | 16 | 935 | 1.712 | 2.188e-07 | 1.094e-07 | pass |
| JeansPy JAX GPU | axisymmetric | 0.7 | 16 | 32 | 948 | 2.55 | 2.375e-10 | 1.188e-10 | pass |
| JeansPy JAX GPU | axisymmetric | 0.7 | 16 | 64 | 1070 | 10.74 | 2.474e-13 | 1.237e-13 | pass |
| JeansPy JAX GPU | axisymmetric | 0.7 | 64 | 16 | 938.6 | 5.936 | 1.185e-07 | 5.923e-08 | pass |
| JeansPy JAX GPU | axisymmetric | 0.7 | 64 | 32 | 923.2 | 10.05 | 1.297e-10 | 6.485e-11 | pass |
| JeansPy JAX GPU | axisymmetric | 0.7 | 64 | 64 | 1106 | 36.05 | 1.323e-13 | 6.617e-14 | pass |
| JeansPy JAX GPU | axisymmetric | 0.7 | 256 | 16 | 962.3 | 23.82 | 1.723e-07 | 8.616e-08 | pass |
| JeansPy JAX GPU | axisymmetric | 0.7 | 256 | 32 | 957.7 | 38.16 | 1.846e-10 | 9.232e-11 | pass |
| JeansPy JAX GPU | axisymmetric | 0.7 | 256 | 64 | 1227 | 143.4 | 1.923e-13 | 9.615e-14 | pass |
| JAM 9.0.2 CPU | jam | 1 | 16 | 24/9/30 | 4.03 | 3.387 | 0.0008487 | 0.0004243 | pass |
| JAM 9.0.2 CPU | jam | 1 | 16 | 32/15/60 | 23.37 | 20.24 | 0.001112 | 0.0005559 | pass |
| JAM 9.0.2 CPU | jam | 1 | 16 | 48/21/90 | 101.4 | 89.37 | 3.265e-05 | 1.632e-05 | pass |
| JAM 9.0.2 CPU | jam | 1 | 64 | 24/9/30 | 4.203 | 2.892 | 0.001072 | 0.000536 | pass |
| JAM 9.0.2 CPU | jam | 1 | 64 | 32/15/60 | 20.58 | 16.54 | 0.001179 | 0.0005893 | pass |
| JAM 9.0.2 CPU | jam | 1 | 64 | 48/21/90 | 95.48 | 95.32 | 5.951e-05 | 2.976e-05 | pass |
| JAM 9.0.2 CPU | jam | 1 | 256 | 24/9/30 | 8.635 | 6.115 | 0.001101 | 0.0005503 | pass |
| JAM 9.0.2 CPU | jam | 1 | 256 | 32/15/60 | 28.75 | 24.42 | 0.001213 | 0.0006065 | pass |
| JAM 9.0.2 CPU | jam | 1 | 256 | 48/21/90 | 116.3 | 116.9 | 5.924e-05 | 2.962e-05 | pass |
| JAM 9.0.2 CPU | jam | 0.7 | 16 | 24/9/30 | 3.014 | 2.14 | 0.004788 | 0.002391 | pass |
| JAM 9.0.2 CPU | jam | 0.7 | 16 | 32/15/60 | 18.89 | 14.67 | 0.002603 | 0.001302 | pass |
| JAM 9.0.2 CPU | jam | 0.7 | 16 | 48/21/90 | 89.42 | 89.75 | 0.0001771 | 8.856e-05 | pass |
| JAM 9.0.2 CPU | jam | 0.7 | 64 | 24/9/30 | 3.728 | 2.94 | 0.004903 | 0.002448 | pass |
| JAM 9.0.2 CPU | jam | 0.7 | 64 | 32/15/60 | 20.89 | 16.8 | 0.004088 | 0.002046 | pass |
| JAM 9.0.2 CPU | jam | 0.7 | 64 | 48/21/90 | 109.5 | 97.51 | 0.0001826 | 9.13e-05 | pass |
| JAM 9.0.2 CPU | jam | 0.7 | 256 | 24/9/30 | 8.293 | 6.222 | 0.005219 | 0.002606 | fail |
| JAM 9.0.2 CPU | jam | 0.7 | 256 | 32/15/60 | 29.57 | 25.25 | 0.003609 | 0.001806 | pass |
| JAM 9.0.2 CPU | jam | 0.7 | 256 | 48/21/90 | 130.5 | 115.2 | 0.0001854 | 9.269e-05 | pass |

## MGE setup and memory

| Tracer q | Requested Gaussian components | Actual components | Fit time [s] | Maximum density error, 1–10,000 pc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | 16 | 2.277 | 0.001863 |
| 1 | 32 | 14 | 1.05 | 0.004841 |
| 1 | 48 | 21 | 0.7796 | 0.0001027 |
| 0.7 | 24 | 15 | 1.555 | 0.005577 |
| 0.7 | 32 | 15 | 1.051 | 0.004616 |
| 0.7 | 48 | 22 | 0.4459 | 0.0008194 |

Each memory entry is the supervisor-observed process high-water mark across the entire role, including model construction, all sizes/settings and JAX compilation. GPU allocations include runtime reservations; these are not incremental per-kernel memory requirements.

| Role | Peak host RSS [GiB] | Observed GPU allocation [GiB] | Role elapsed time [s] |
| --- | ---: | ---: | ---: |
| JeansPy NumPy | 0.753 | 0.000 | 483.717 |
| JeansPy JAX CPU | 0.385 | 0.000 | 102.738 |
| JeansPy JAX GPU | 0.924 | 0.314 | 53.046 |
| JAM 9.0.2 CPU | 0.229 | 0.000 | 23.527 |

## Independent reference and limits

For $q=1$, the reference is the exact expression $3\pi GM/(64\sqrt{a^2+x^2+y^2})$. For $q=0.7$, use dimensionless $a=GM=1$, $A=1+R^2$, $T=A+z^2$, $C=3/(4\pi q)$ and $w=(1-q^2)A/T$. Direct integration of the vertical Jeans equation gives

```{math}
p\equiv\nu\sigma_z^2=\frac{Cq^5}{6T^3}\,{}_2F_1(5/2,3;4;w).
```

Its analytic radial derivative, together with $\nu\overline{v_\phi^2}=p+R\partial_Rp+R\nu\partial_R\Phi$, gives the LOS integrand $p+\sin^2(i)x^2[(\partial_Rp)/R+\nu/T^{3/2}]$. SciPy adaptive quadrature integrates over the infinite LOS, then divides by the exact projected tracer density and restores the factor $GM/a$. This calculation uses no JeansPy force or numerical-integration helper. Pressure and radial derivative are checked against separate direct integrals at eight intrinsic points; both projection tolerances are evaluated at every sky position. The spherical numerical projection is additionally checked against the exact formula. All reference checks pass the fixed $10^{-8}$ relative criteria; the quadrature error estimates and every prediction remain in the reference record.

The experiment covers two smooth models with spherical gravity, zero meridional anisotropy, one host and one GPU. It does not establish accuracy for cusps, flattened halos, finite cutoffs, nonzero anisotropies or a whole prior domain. Higher quadrature order has a measured cost; low-order agreement here does not justify a default change. No runtime scientific definition or numerical default was changed.

## Reproduction and source data

CPU: AMD Ryzen Threadripper 3990X 64-Core Processor, logical CPUs 0–7, BLAS/OpenMP threads set to one. JAX CPU/GPU use float64. GPU: NVIDIA GeForce RTX 3090 (24 GiB, driver 595.84); process device cuda:0. Dependency versions, executable paths, CPU affinity, protocol/runner/runtime hashes, every repetition, warnings and observed memory are retained in the JSON and supervisor files.

Download the {download}`complete timing table <../../../validation/release/los-benchmark/summary.csv>`, {download}`frozen protocol <../../../validation/release/los_benchmark_protocol.json>` and {download}`independent reference <../../../validation/release/los-benchmark/reference.json>`. The per-role raw outputs are {download}`NumPy <../../../validation/release/los-benchmark/numpy.json>`, {download}`JAX CPU <../../../validation/release/los-benchmark/jax-cpu.json>`, {download}`JAX GPU <../../../validation/release/los-benchmark/jax-gpu.json>` and {download}`JAM <../../../validation/release/los-benchmark/jam.json>`.

Regenerate this page and its figures without numerical experiments:

```bash
python scripts/render_release_los.py
```

For a new experiment, use the committed protocol with `scripts/benchmark_release_los.py --role reference --output <new-reference.json>`, followed by roles `numpy`, `jax-cpu`, `jax-gpu` and `jam`, each with `--reference <new-reference.json>` and a fresh output path. Launch through the cumulative-budget supervisor with the role's frozen time/memory bound, `taskset -c 0-7`, `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. Configure `JEANSPY_JAX_PLATFORM=cpu` or `gpu` and `JEANSPY_JAX_ENABLE_X64=true` before the corresponding Python process starts. JAM uses a separate environment with `jampy==9.0.2` and `mgefit==6.2.6`. Exact commands and dependency versions for these runs are in the retained supervisor/role records. The current campaign uses one existing 24-hour ledger; this page does not authorize resetting or extending it.
