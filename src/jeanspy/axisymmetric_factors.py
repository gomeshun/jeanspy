"""Finite-distance J/D integrals for a spheroidally truncated Zhao halo.

The aperture is a circular cone on the observer's sky, not a spherical volume
or a distant-observer cylinder. The same ellipsoidal cutoff used in the force
defines the emitting mass. Fixed orders require parameter-specific refinement.
"""
from __future__ import annotations

import numpy as np
from scipy.constants import parsec

from ._classical.jfactor import C_J, kg_eV, solar_mass_kg
from .axisymmetric import ZhaoHalo, _inclination, _positive, _rule

__all__ = ["jfactor", "dfactor", "C_J", "C_D"]

#: Convert a numerical J factor from Msun^2/pc^5 to GeV^2/cm^5.
C_J: float

#: Convert a numerical D factor from Msun/pc^2 to GeV/cm^2.
C_D = (solar_mass_kg * kg_eV / 1e9) / (parsec * 100)**2


def _factor(halo, dist_pc, roi_deg, inclination, power, n_mu, n_phi, n_radial):
    if not isinstance(halo, ZhaoHalo):
        raise TypeError("halo must be a ZhaoHalo")
    _positive("dist_pc", dist_pc)
    _inclination(inclination)
    if not np.isscalar(roi_deg) or not np.isfinite(roi_deg) or not 0 <= roi_deg < 90:
        raise ValueError("roi_deg must be in [0, 90) degrees")
    if not np.isfinite(halo.r_t_pc):
        raise ValueError("J/D factors require an explicit finite ellipsoidal r_t_pc")
    if dist_pc <= halo.r_t_pc * max(1., halo.Q):
        raise ValueError("Observer must be outside the halo: dist_pc > r_t_pc * max(1, Q)")
    exponent = 3-power*halo.gamma
    if exponent <= 0:
        raise ValueError("The central aperture integral diverges: require power * gamma < 3")
    u_mu, w_mu = _rule(n_mu)
    _rule(n_phi)  # same order validation; azimuth uses a periodic trapezoid rule
    u, w = _rule(n_radial)
    if roi_deg == 0:
        return 0.

    si, ci = np.sin(inclination), np.cos(inclination)
    theta = np.deg2rad(roi_deg)
    st, ct = np.sin(theta), np.cos(theta)
    phi = 2*np.pi*(np.arange(n_phi)+.5)/n_phi
    cp, sp = np.cos(phi), np.sin(phi)
    logu = np.log(u)
    profile_power = power*(halo.beta-halo.gamma)/halo.alpha
    total = 0.
    for mu, wm in zip(2*u_mu-1, 2*w_mu):
        equatorial = np.sqrt(1-mu*mu)
        ex, ey, ez = equatorial*cp, equatorial*sp, halo.Q*mu
        los = ey*si+ez*ci
        transverse = np.hypot(ex, ey*ci-ez*si)
        norm2 = equatorial**2+(halo.Q*mu)**2
        # A point at spheroidal radius m is within the cone iff
        # m * (transverse*cos(theta) - los*sin(theta)) <= D*sin(theta).
        # s^2 = D^2 + 2*D*m*los + m^2*norm2 is the observer distance.
        denominator = transverse*ct-los*st
        cone_radius = dist_pc*st/np.where(denominator > 0, denominator, 1.)
        mmax = np.where(denominator > 0, np.minimum(cone_radius, halo.r_t_pc), halo.r_t_pc)
        xmax = mmax/halo.r_s
        logc = np.log(np.minimum(xmax, 1.))[:, None]
        logx = logc+4/exponent*logu
        radius = halo.r_s*np.exp(logx)
        distance_weight = 1+2*(radius/dist_pc)*los[:, None]+(radius/dist_pc)**2*norm2
        shape = np.exp(-profile_power*np.logaddexp(0., halo.alpha*logx))
        # Cusp regularization cancels all singular powers analytically. In
        # particular, no central core or radius floor is introduced as gamma
        # approaches the integrability boundary.
        inner = 4/exponent*np.exp(exponent*logc[:, 0])*np.sum(w*u**3*shape/distance_weight, axis=-1)

        length = np.log(np.maximum(xmax, 1.))
        logx = length[:, None]*u
        radius = halo.r_s*np.exp(logx)
        distance_weight = 1+2*(radius/dist_pc)*los[:, None]+(radius/dist_pc)**2*norm2
        shape = np.exp(exponent*logx-profile_power*np.logaddexp(0., halo.alpha*logx))
        outer = length*np.sum(w*shape/distance_weight, axis=-1)
        total += wm*2*np.pi/n_phi*np.sum(inner+outer)
    return float(halo.Q*halo.rho_s**power*halo.r_s**3/dist_pc**2*total)


def jfactor(halo, dist_pc, roi_deg, *, inclination=np.pi/2,
            n_mu=96, n_phi=96, n_radial=128):
    r"""Return integral rho^2 ds dOmega in GeV^2 cm^-5 (gamma must be < 1.5).

    Notes
    -----
    **Inputs and units.** halo is ZhaoHalo; ``dist_pc`` is scalar observer
    distance (pc); ``roi_deg`` is scalar cone half-angle; inclination is
    radians; ``n_mu``/``n_phi``/``n_radial`` are integers >=16.

    **Returns and shape.** Nonnegative Python float in GeV^2 cm^-5; zero
    aperture gives zero.

    **Validity.** Require explicit finite ``r_t_pc``, observer distance >
    ``r_t_pc``\*max(1,Q), ``0 <= roi_deg < 90`` and gamma<1.5 (finite central
    annihilation integral). ``n_phi`` uses a periodic rule; refine all orders.

    **Errors.** Invalid geometry/domain/order raises ValueError; wrong halo type
    raises TypeError.

    **Backend.** NumPy/SciPy CPU postprocessing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_factors.py``
    """
    return C_J*_factor(halo, dist_pc, roi_deg, inclination, 2, n_mu, n_phi, n_radial)


def dfactor(halo, dist_pc, roi_deg, *, inclination=np.pi/2,
            n_mu=96, n_phi=96, n_radial=128):
    r"""Return integral rho ds dOmega in GeV cm^-2.

    Notes
    -----
    **Inputs and units.** halo is ZhaoHalo; ``dist_pc`` is scalar observer
    distance (pc); ``roi_deg`` is scalar cone half-angle; inclination is
    radians; ``n_mu``/``n_phi``/``n_radial`` are integers >=16.

    **Returns and shape.** Nonnegative Python float in GeV cm^-2; zero aperture
    gives zero.

    **Validity.** Require explicit finite ``r_t_pc``, observer distance >
    ``r_t_pc``\*max(1,Q), ``0 <= roi_deg < 90`` and gamma<2 under the ZhaoHalo
    constructor domain. ``n_phi`` uses a periodic rule; refine all orders.

    **Errors.** Invalid geometry/domain/order raises ValueError; wrong halo type
    raises TypeError.

    **Backend.** NumPy/SciPy CPU postprocessing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_factors.py``
    """
    return C_D*_factor(halo, dist_pc, roi_deg, inclination, 1, n_mu, n_phi, n_radial)
