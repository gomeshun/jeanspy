# Ullio & Valli J-factor geometry in JeansPy

JeansPy exposes two related classical J-factor methods:

- `DMModel.jfactor_ullio2016(...)`: the full finite-ROI Ullio & Valli (2016) geometry.
- `DMModel.jfactor_ullio2016_simple(...)`: a spherical-aperture approximation.

Let

\[
R_{\max}=D\sin\theta_{\max}
\]

be the projected aperture radius and let `r_t` be the halo truncation radius, so that the model is assumed to have no density for `r > r_t`.

## Full finite-ROI method

`jfactor_ullio2016(...)` evaluates the Ullio & Valli finite-aperture geometry. When `R_max < r_t`, it includes the contribution from shells with

\[
R_{\max} < r < r_t
\]

whose projected radius still lies inside the observed aperture. This is the recommended reference calculation for a general finite ROI.

## Simple spherical-aperture method

The generic `jfactor_ullio2016_simple(...)` integrates

\[
J_{\rm simple}=\frac{4\pi}{D^2}
\int_0^{\min(R_{\max},r_t)} r^2\rho^2(r)\,dr.
\]

Its interpretation depends on the relative sizes of the aperture and the truncated halo:

- **If `R_max >= r_t`**, the aperture contains the whole truncated halo. Then `min(R_max, r_t) = r_t`, and the generic simple expression coincides with the small-angle Ullio & Valli Eq. B.10 after identifying their halo radius `\mathcal R` with `r_t`.
- **If `R_max < r_t`**, the method drops the projected contribution from shells with `r > R_max`. In this regime it is a spherical-aperture approximation and should not be identified with the full finite-ROI result.

Thus `min(R_max, r_t)` is intentional: it is exact for the radial support of a truncated halo once the aperture encloses the full halo, but it is only an approximation when the aperture cuts through the halo.

## NFW override

`NFWModel.jfactor_ullio2016_simple(...)` has the same spherical truncation semantics, using `min(R_max, r_t)`, but it additionally retains the existing analytic finite-distance correction term. Its leading small-angle term reduces to the Eq. B.10 expression when `R_max >= r_t`.

For analyses where the distinction matters, prefer `jfactor_ullio2016(...)` and use `jfactor_ullio2016_simple(...)` as a fast approximation or cross-check.
