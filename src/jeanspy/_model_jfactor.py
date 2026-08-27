"""Dark-matter profiles and J-factor geometry for the classical backend."""

from __future__ import annotations

from abc import abstractmethod
from scipy.constants import parsec, physical_constants
from scipy.integrate import quad
from scipy.special import beta, betainc

import numpy as np
from numpy import log, log1p, pi, power

from ._model_base import Model


GMsun_m3s2 = 1.32712440018e20
R_trunc_pc = 1866.

kg_eV = 1./physical_constants["electron volt-kilogram relationship"][0]
im_eV = 1./physical_constants["electron volt-inverse meter relationship"][0]
solar_mass_kg = 1.9884e30
C0 = (solar_mass_kg*kg_eV)**2*((1./parsec)*im_eV)**5
C1 = (1e9)**2 * (1e2*im_eV)**5
C_J = C0/C1


def _ullio2016_weight(r_pc, s_pc, t_pc, dist_pc):
    """Return the Ullio & Valli (2016) shell weight ``W(r; s, t)``."""
    try:
        r, s, t, dist = np.broadcast_arrays(
            np.asarray(r_pc, dtype=float),
            np.asarray(s_pc, dtype=float),
            np.asarray(t_pc, dtype=float),
            np.asarray(dist_pc, dtype=float),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Ullio geometry arguments must be real numbers.") from exc

    if (
        np.any(~np.isfinite(r))
        or np.any(~np.isfinite(s))
        or np.any(~np.isfinite(t))
        or np.any(~np.isfinite(dist))
        or np.any(r < 0)
        or np.any(s < 0)
        or np.any(t < s)
        or np.any(t > r)
        or np.any(dist <= r)
    ):
        raise ValueError("Require 0 <= s <= t <= r < dist for the Ullio shell weight.")

    r2_minus_s2 = np.maximum((r - s) * (r + s), 0.0)
    r2_minus_t2 = np.maximum((r - t) * (r + t), 0.0)
    dist2_minus_s2 = np.maximum((dist - s) * (dist + s), 0.0)
    dist2_minus_t2 = np.maximum((dist - t) * (dist + t), 0.0)

    # This is an algebraically stable form of
    # asinh(sqrt((r^2-s^2)/(D^2-r^2))) -
    # asinh(sqrt((r^2-t^2)/(D^2-r^2))).
    numerator = (t - s) * (t + s)
    denominator = (
        np.sqrt(r2_minus_s2 * dist2_minus_t2)
        + np.sqrt(r2_minus_t2 * dist2_minus_s2)
    )
    argument = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    return np.where(r > 0, (r / dist) * np.arcsinh(argument), 0.0)


def _ullio2016_inner_weight(r_pc, dist_pc):
    """Return ``W(r; 0, r)`` with a small-distance series limit."""
    r = np.asarray(r_pc, dtype=float)
    dist = np.asarray(dist_pc, dtype=float)
    if (
        np.any(~np.isfinite(r))
        or np.any(~np.isfinite(dist))
        or np.any(r < 0)
        or np.any(dist <= r)
    ):
        raise ValueError("Require 0 <= r < dist for the Ullio shell weight.")

    q = r / dist
    q2 = q * q
    series = q2 * (1.0 + q2 / 3.0 + q2 * q2 / 5.0 + q2 * q2 * q2 / 7.0)
    return np.where(q2 < 1.0e-8, series, q * np.arctanh(q))

class DMModel(Model):
    name = "DM Model"


    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.roi_deg_max_warning = 1.0  # maximum angle for small-angle approximations

    @abstractmethod
    def mass_density_3d(self,r_pc):
        pass

    def enclosed_mass(self, r_pc):
        """Return the mass enclosed within ``r_pc``.

        ``enclosure_mass`` is retained on the classical concrete models for
        compatibility with the original API.  This spelling is shared with
        the NumPyro backend.
        """
        return self.enclosure_mass(r_pc)

    def _validate_jfactor_inputs(self, dist_pc, roi_deg, *, full=False, small_angle=False):
        try:
            dist_pc = np.asarray(dist_pc, dtype=float)
            roi_deg = np.asarray(roi_deg, dtype=float)
            r_t_pc = np.asarray(self.params["r_t_pc"], dtype=float)
            dist_pc, roi_deg, r_t_pc = np.broadcast_arrays(dist_pc, roi_deg, r_t_pc)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "J-factor evaluation requires a finite truncation radius r_t_pc."
            ) from exc

        if (
            np.any(~np.isfinite(dist_pc))
            or np.any(~np.isfinite(roi_deg))
            or np.any(~np.isfinite(r_t_pc))
            or np.any(dist_pc <= 0)
            or np.any(roi_deg <= 0)
            or np.any(r_t_pc <= 0)
        ):
            raise ValueError("dist_pc, roi_deg, and r_t_pc must be finite and positive.")

        if small_angle and np.any(roi_deg > self.roi_deg_max_warning):
            raise ValueError(
                f"Small-angle J-factor approximations require roi_deg <= "
                f"{self.roi_deg_max_warning} degrees."
            )

        if full:
            if np.any(dist_pc <= r_t_pc):
                raise ValueError(
                    "The observer must be outside the truncated halo: dist_pc > r_t_pc."
                )
            if np.any(roi_deg > 90.0):
                raise ValueError(
                    "The full Ullio geometry supports apertures no larger than 90 degrees."
                )

        return dist_pc, roi_deg, r_t_pc

    def assert_roi_is_enough_small(self,roi_deg):
        try:
            roi_deg = np.asarray(roi_deg, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("roi_deg must be a finite positive number.") from exc
        if np.any(~np.isfinite(roi_deg)) or np.any(roi_deg <= 0):
            raise ValueError("roi_deg must be a finite positive number.")
        if np.any(roi_deg > self.roi_deg_max_warning):
            raise ValueError(
                f"Small-angle J-factor approximations require roi_deg <= "
                f"{self.roi_deg_max_warning} degrees."
            )

    def jfactor_ullio2016_simple(self, dist_pc, roi_deg=0.5):
        """Calculate the spherical-aperture approximation to the J-factor.

        This compatibility method integrates ``4*pi*r**2*rho(r)**2 / dist_pc**2``
        up to ``min(R_max, r_t_pc)``, where ``R_max = dist_pc*sin(roi_deg)``.
        It omits the projected contribution of shells with ``r > R_max``; the
        full finite-ROI Ullio geometry is provided by :meth:`jfactor_ullio2016`.
        """
        dist_pc, roi_deg, r_t_pc = self._validate_jfactor_inputs(
            dist_pc, roi_deg, small_angle=True
        )
        if dist_pc.ndim != 0 or roi_deg.ndim != 0 or r_t_pc.ndim != 0:
            raise ValueError("The J-factor methods require scalar model geometry.")

        dist_pc = float(dist_pc)
        roi_deg = float(roi_deg)
        r_t_pc = float(r_t_pc)
        r_max_pc = min(dist_pc * np.sin(np.deg2rad(roi_deg)), r_t_pc)

        def integrand(r_pc):
            return r_pc**2 * float(np.asarray(self.mass_density_3d(r_pc)))**2

        integ, _ = quad(integrand, 0.0, r_max_pc, epsabs=0.0, epsrel=1.0e-8, limit=300)
        return C_J * 4.0 * np.pi / dist_pc**2 * integ

    def jfactor_ullio2016(self, dist_pc, roi_deg=0.5):
        """Calculate the full finite-ROI Ullio & Valli (2016) J-factor.

        The radial integral uses the finite halo radius ``r_t_pc`` and retains
        the projected contribution from shells outside the aperture.
        """
        dist_pc, roi_deg, r_t_pc = self._validate_jfactor_inputs(
            dist_pc, roi_deg, full=True
        )
        if dist_pc.ndim != 0 or roi_deg.ndim != 0 or r_t_pc.ndim != 0:
            raise ValueError("The J-factor methods require scalar model geometry.")

        dist_pc = float(dist_pc)
        roi_deg = float(roi_deg)
        r_t_pc = float(r_t_pc)
        r_max_pc = dist_pc * np.sin(np.deg2rad(roi_deg))
        r_inner_pc = min(r_max_pc, r_t_pc)

        def inner_integrand(r_pc):
            if r_pc == 0.0:
                return 0.0
            rho = float(np.asarray(self.mass_density_3d(r_pc)))
            return rho**2 * float(_ullio2016_inner_weight(r_pc, dist_pc))

        integ, _ = quad(
            inner_integrand,
            0.0,
            r_inner_pc,
            epsabs=0.0,
            epsrel=1.0e-8,
            limit=300,
        )

        if r_max_pc < r_t_pc:
            outer_width_pc = r_t_pc - r_max_pc

            def outer_integrand(u):
                r_pc = r_max_pc + outer_width_pc * u**2
                rho = float(np.asarray(self.mass_density_3d(r_pc)))
                weight = _ullio2016_weight(r_pc, 0.0, r_max_pc, dist_pc)
                return 2.0 * outer_width_pc * u * rho**2 * float(weight)

            outer, _ = quad(
                outer_integrand,
                0.0,
                1.0,
                epsabs=0.0,
                epsrel=1.0e-8,
                limit=300,
            )
            integ += outer

        return C_J * 4.0 * np.pi * integ


class ZhaoModel(DMModel):
    """ DM model whose 3D (deprojected) density is given by the Zhao model,
    which is a generalization of the NFW model.

    The Zhao model is given by the following formula:
    \rho(r) = \rho_s (r/r_s)^{-g} (1+(r/r_s)^a)^{-(b-g)/a}

    where r_s is the scale radius, \rho_s is the scale density, a is the transition sharpness, b is the outer slope and g is the inner slope.
    NFW model is a special case of the Zhao model with (a,b,g) = (1,3,1).
    """
    name = "Zhao Model"
    required_param_names = ['rs_pc','rhos_Msunpc3','a','b','g','r_t_pc']
    required_models = {}


    def mass_density_3d(self,r_pc):
        rs_pc, rhos_Msunpc3,a,b,g = self.params.rs_pc, self.params.rhos_Msunpc3, self.params.a, self.params.b,self.params.g
        x = r_pc/rs_pc
        return rhos_Msunpc3*power(x,-g)*power(1+power(x,a),-(b-g)/a)

    def enclosure_mass(self,r_pc):
        rs_pc, rhos_Msunpc3,a,b,g = self.params.rs_pc, self.params.rhos_Msunpc3, self.params.a, self.params.b,self.params.g
        r_t_pc = self.params.r_t_pc

        r_pc_trunc = np.where(r_pc>r_t_pc,r_t_pc,r_pc)

        # The beta/betainc form becomes indeterminate at the exact NFW limit
        # (a, b, g) = (1, 3, 1), even though the enclosed mass is finite.
        if np.isclose(a, 1.0, atol=1e-7, rtol=0.0) and np.isclose(b, 3.0, atol=1e-7, rtol=0.0) and np.isclose(g, 1.0, atol=1e-7, rtol=0.0):
            x_nfw = r_pc_trunc/rs_pc
            return (4.*pi*rs_pc**3 * rhos_Msunpc3) * (log1p(x_nfw) - x_nfw/(1+x_nfw))

        x = power(r_pc_trunc/rs_pc,a)
        argbeta0 = (3-g)/a
        argbeta1 = (b-3)/a

        return (4.*pi*rs_pc**3 * rhos_Msunpc3/a) * beta(argbeta0,argbeta1) * betainc(argbeta0,argbeta1,x/(1+x))


class NFWModel(DMModel):
    name = "NFW Model"
    required_param_names = ['rs_pc','rhos_Msunpc3','r_t_pc']
    required_models = {}


    def mass_density_3d(self,r_pc):
        rs_pc, rhos_Msunpc3 = self.params.rs_pc, self.params.rhos_Msunpc3
        x = r_pc/rs_pc
        return rhos_Msunpc3/x/(1+x)**2

    def enclosure_mass(self,r_pc):
        threshold = 1e-7  # threshold to avoid underflow
        rs_pc, rhos_Msunpc3 = self.params.rs_pc, self.params.rhos_Msunpc3
        r_t_pc = self.params.r_t_pc
        if isinstance(r_pc,np.ndarray):
            r_pc_trunc = np.where(r_pc>r_t_pc,r_t_pc,r_pc)
        else:
            r_pc_trunc = min(r_pc,r_t_pc)
        x = r_pc_trunc/rs_pc
        is_small = x < threshold
        # NOTE: (1/(1+x)-1 + log(1+x)) = B(2,0,x/(1+x)),
        # but scipy.special.betainc and scipy.special.beta are useless because of their diversence.
        # Therefore we use another expression in the following calculation.
        # Note that the element specification is relatively slow, thus we calculate all elements first and then modify overflowed ones.
        ret = (1/(1+x)-1 + log(1+x))  # NOTE:  underflow occurs when x<<1.
        ret = np.where(is_small, np.asarray(x)**2/2, ret)  # Series expansion up to second order
        return (4.*pi*rs_pc**3 * rhos_Msunpc3) * ret

    def jfactor_ullio2016_simple(self,dist_pc,roi_deg=0.5):
        """Calculate the spherical-aperture J-factor approximation.

        The projected contribution from shells with ``r > R_max`` is omitted.
        """
        dist_pc, roi_deg, r_t_pc = self._validate_jfactor_inputs(
            dist_pc, roi_deg, small_angle=True
        )
        roi_pc = dist_pc*np.sin(np.deg2rad(roi_deg))
        rs_pc, rhos_Msunpc3 = self.params.rs_pc, self.params.rhos_Msunpc3
        r_max_pc = np.minimum(roi_pc, r_t_pc)
        c_max = r_max_pc/rs_pc
        j = C_J * 4 * pi * rs_pc**3 * rhos_Msunpc3**2 / dist_pc**2  # normalization
        j *= (1-1/(1+c_max)**3)/3 + ((rs_pc/dist_pc)**2 * c_max**3/(1+c_max)**3)/9  # approximation of W(r,0,r) upto second leading order
        return j

    def jfactor_evans2016(self,dist_pc,roi_deg=0.5):
        """J-factor fitting function given by https://arxiv.org/pdf/1604.05599.pdf
        Note that this formula causes the cancelation of significant digits
        """
        self.assert_roi_is_enough_small(roi_deg)
        def func_x(s):
            epsilon = 1e-8
            s = np.atleast_1d(s)
            assert np.all(s>=0)
            ret = np.nan * np.ones_like(s)
            cond_1 = s<1-epsilon
            cond_2 = 1+epsilon<s
            cond_3 = np.abs(1-s) <= epsilon
            ret[cond_1] = np.arccosh(1/s[cond_1])/np.sqrt(1-s[cond_1]**2)
            ret[cond_2] = np.arccos(1/s[cond_2])/np.sqrt(s[cond_2]**2-1)
            ret[cond_3] = 1 - (2*(s[cond_3]-1))/3 + 7/15 * (s[cond_3]-1)**2
            return ret

        roi_pc = dist_pc * np.deg2rad(roi_deg)
        rs_pc, rhos_Msunpc3 = self.params.rs_pc, self.params.rhos_Msunpc3
        r_t_pc = self.params.r_t_pc
        r_max_pc = np.min([np.ones_like(r_t_pc)*roi_pc,r_t_pc],axis=0)
        y =  r_max_pc / rs_pc
        delta = 1 - y**2
        coeff_evans = (2*y*(7*y-4*y**3+3*pi*delta**2) + 6*(2*delta**3-2*delta-y**4)*func_x(y)) / 6 / delta**2
        epsilon = 1e-8
        coeff_evans[np.abs(1-y)<epsilon] = (np.pi - 38/15) + (-(64/21) + np.pi)*(y-1)
        j = C_J * 2 * pi * rhos_Msunpc3**2 * rs_pc**3 / dist_pc**2 * coeff_evans
        return j
