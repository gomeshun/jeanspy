# Theory and numerical methods

## Spherical Jeans equations

For a steady, spherical, nonrotating collisionless tracer, define
$\beta(r)=1-\sigma_\theta^2/\sigma_r^2$ with
$\sigma_\theta^2=\sigma_\phi^2$. The second-moment equation is

```{math}
:label: spherical-jeans
\frac{d(\nu\sigma_r^2)}{dr}+\frac{2\beta}{r}\nu\sigma_r^2
=-\nu\frac{GM(r)}{r^2}.
```

The boundary condition is vanishing radial pressure at infinity. Projection
gives

```{math}
\Sigma(R)\sigma_{\rm los}^2(R)=2\int_R^\infty
\left(1-\beta(r)\frac{R^2}{r^2}\right)
\frac{\nu(r)\sigma_r^2(r)r\,dr}{\sqrt{r^2-R^2}}.
```

The classical implementation uses SciPy integration for intrinsic moments
and double-exponential quadrature for the kernel LOS solver. The JAX
implementation exposes kernel and direct routes with fixed quadrature.
The route, node counts and infinity transformation are part of the numerical
model and must be recorded. Agreement of two backends is a consistency check;
an analytic solution or independent integral supplies stronger validation.

## Axisymmetric Jeans equations

JeansPy implements the stationary, cylindrically aligned model of
[Hayashi & Chiba (2015)](https://arxiv.org/abs/1507.07620) and
[Hayashi et al. (2016)](https://arxiv.org/abs/1603.08046). The stellar and halo
axes coincide, cross moments vanish, and $\beta_z=1-\overline{v_z^2}/\overline{v_R^2}$
is constant. With $P=\nu\overline{v_z^2}$,

```{math}
P(R,z)=\int_{|z|}^\infty\nu(R,z')\Phi_z(R,z')\,dz',\qquad
\overline{v_R^2}=\frac{P}{(1-\beta_z)\nu},\qquad
\overline{v_\phi^2}=\frac{P+R\,\partial_RP}{(1-\beta_z)\nu}+R\Phi_R.
```

Force, vertical-pressure and LOS integrals use fixed Gauss–Legendre rules.
Rational maps cover infinite vertical and LOS ranges. The halo may have a
finite spheroidal cutoff; differentiation of the force integral then includes
the moving integration endpoint. The [axisymmetric guide](guides/axisymmetric.md)
details the force equations and parameter domains.

Neither finite positive moments at observed positions nor successful MCMC
prove that a positive distribution function exists everywhere. Rotation,
tilted velocity ellipsoids, triaxiality, stellar self-gravity, PSF convolution
and spatial-bin averaging are outside the current axisymmetric model.

## Likelihood and interpretation

The standard unbinned likelihood conditions on positions and independent
Gaussian velocity errors. It approximates the velocity distribution by a
Gaussian with the predicted second moment:

```{math}
\log L=\sum_i\log\mathcal N\!\left(v_i\mid v_{\rm sys},
\sqrt{\sigma_{\rm los}^2(\boldsymbol x_i)+\epsilon_i^2}\right).
```

This does not imply that Jeans equations uniquely determine the full velocity
distribution. Membership cuts, binaries, contaminants, equilibrium and the
choice of tracer are scientific assumptions that require sensitivity checks.
Priors are defined in the explicitly named sampling coordinates. A uniform
prior in `log10_rs_pc` is log-uniform in physical radius; no extra Jacobian
is added to redefine it as a uniform physical-radius prior.
