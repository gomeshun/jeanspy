# Segue 1 precision warning: resolved for the revised plots

The two archived fixed-point failures were caused by amplified floating-point error in nearly linearly dependent Fisher-score columns. At the worse point the normalized minimum/maximum singular-value ratio is about 9×10⁻¹⁵. Double precision derivative rounding is amplified when computing the information volume. Scaling the columns alone cannot recover lost digits.

An experiment-local `PrecisionSafeMarginal` retains the frozen likelihood, seven-coordinate joint Jeffreys measure, finite systemic-velocity integral, all original parameter bounds, clipping, and quadrature. It chooses binary128 arithmetic when the normalized singular-value ratio falls below 10⁻⁷. This threshold chooses arithmetic only: it neither rejects proposals nor truncates singular directions. Forward derivatives are evaluated independently in C++ with GNU libquadmath, and two volume algorithms (Householder and twice-reorthogonalized Gram-Schmidt) agree at the original failed points to better than 5×10⁻²² in log volume. The wrapper is an emcee-compatible host evaluator; it is not a differentiable NumPyro/NUTS replacement. Future sampling with it requires a new sampling identity/output.

The original failed checks and original protocol remain intact. The follow-up results are in `resolution.json`:

- All 66 original fixed points (including generating points) now have CPU/GPU maximum difference 1.64×10⁻¹⁰, below the unchanged 10⁻⁶ threshold.
- Independent extended/binary128 reference agreement across those points is better than 8.25×10⁻¹¹.
- At both formerly failed points, doubled quadrature and a tenfold endpoint satisfy the original 0.05 log-prior and 0.005 relative-variance tolerances. At the worse point the endpoint shift is about 0.0346 in log prior; this is below 0.05, not evidence of 10⁻⁶ quadrature accuracy. These are different numerical gates.
- GPU reevaluation of 33,967 saved posterior points (uniform random selections per ensemble plus low/high log-density and coordinate extrema) agrees with stored densities to 1.14×10⁻¹³. None requires high-precision routing; the minimum Jeffreys singular ratio is 1.48×10⁻⁶.
- CPU/GPU comparison of a deterministic 2,276-point subset, emphasizing the smallest singular ratios and lowest densities, has maximum log-density difference 3.87×10⁻¹¹.
- Independent extended-precision calculations on 64 Jeffreys posterior points, including the worst-conditioned and low-density tails, agree to 3.01×10⁻¹¹. The worst four points in each case also pass doubled quadrature. MAP/MLE coordinates were checked by the independent evaluator.

No MCMC resampling, reweighting of the published plots, support restriction, tolerance relaxation, or source-chain mutation was performed. The validation supports retaining these saved samples for these fixed-mock comparison plots. It is a finite audit, not an all-support error bound or repeated-mock calibration; it does not demonstrate superior core/cusp discrimination. The old source assessment still records the status at campaign completion, not this addendum.

The attempted full CPU audit was operationally bounded at 600 s. It did not finish all targets and is not counted as a completed audit. The separately completed compact CPU audit, full selected-point GPU audit, and independent reference checks form the reported evidence. Failed development attempts (library-name collision; JSON serialization of the infinite cutoff) remain archived, and successful retries are separately identified.

## Reproduction and provenance

Compile reference.cpp with `g++ -O3 -std=c++17 -fPIC -shared reference.cpp -o libreference.so -lquadmath`; compile the independent extended version with `-DEXTENDED` and output libreference_extended.so. Use the saved Segue environment, explicit CPU/GPU float64 settings, and single BLAS/OpenMP threads. `prepare_audit.py` freezes point selection. `diagnose.py`, `refine_reference.py`, `check_fixed.py`, `audit_backend.py`, `audit_cpu_compact.py`, `audit_reference.py`, and `summarize.py` retain inputs/results. Existing fixed-point files may be reused after interrupted report serialization; remove only copies in a fresh audit directory for a complete fresh reproduction. `resolution.json` contains the source-protocol hash. The campaign's immutable source and original HDF hashes are referenced by the plot band metadata.

[GNU libquadmath API](https://gcc.gnu.org/onlinedocs/libquadmath/) documents the reference arithmetic functions. The scientific formulas come from the frozen local target, not a new prior definition.
