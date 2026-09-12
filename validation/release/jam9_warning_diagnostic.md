# Follow-up to the frozen JAM 9 Plummer validation

The first execution of protocol `jam9-plummer-isotropic-v1`, frozen in commit
`51efc65`, passed all numerical gates but emitted two `invalid value encountered
in log` warnings. The original results remain in `jam9_plummer_v1.json`.

Before this follow-up runs, its sole question is fixed: which intrinsic
resolution and source line emits this warning? Evaluate the spherical Plummer
intrinsic case at the original three spectral resolutions and at the finest
resolution with supplied analytic spatial derivatives. Preserve warning
locations and all returned moments. Do not change models, coordinates, grids,
or thresholds. This diagnostic does not replace the original experiment and
adds no accuracy claim. Limit: 120 seconds, one CPU thread, 16 GiB address space.

Run `scripts/diagnose_jam9_warnings.py --output <new-file.json>` in the pinned
JAM 9 environment with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. If the warning
does not recur, retain that outcome rather than expanding the investigation
without recording a new question.
