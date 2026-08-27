"""Physical profile components for the classical NumPy/SciPy backend."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from scipy.integrate import quad
from scipy.special import beta, betainc, hyp2f1, k0

from ..dequad import dequad
from .core import Model
from .jfactor import C_J, _ullio2016_inner_weight, _ullio2016_weight


class StellarModel(Model):
    """Base class for projected/deprojected stellar-density models."""

    name = "stellar Model"
    required_models = {}

    def density(self, distance_from_center, dimension):
        if dimension == "2d":
            return self.density_2d(distance_from_center)
        if dimension == "3d":
            return self.density_3d(distance_from_center)
        raise ValueError("dimension must be either '2d' or '3d'.")

    def density_2d_truncated(self, R_pc, R_trunc_pc):
        r"""Return the normalized 2-D density truncated at ``R_trunc_pc``.

        The normalization satisfies

        .. math::

            \int_0^{R_\mathrm{trunc}} 2\pi R\,\Sigma_\mathrm{trunc}(R)\,dR = 1.
        """
        return self.density_2d(R_pc) / self.cdf_R(R_trunc_pc)

    @abstractmethod
    def density_2d(self, R_pc):
        raise NotImplementedError

    @abstractmethod
    def density_3d(self, r_pc):
        raise NotImplementedError


class PlummerModel(StellarModel):
    name = "Plummer Model"
    required_param_names = ["re_pc"]
    required_models = {}

    def density_2d(self, R_pc):
        re_pc = self.params.re_pc
        return 1.0 / (1.0 + (R_pc / re_pc) ** 2) ** 2 / np.pi / re_pc**2

    def logdensity_2d(self, R_pc):
        re_pc = self.params.re_pc
        return (
            -2.0 * np.log1p((R_pc / re_pc) ** 2)
            - np.log(np.pi)
            - 2.0 * np.log(re_pc)
        )

    def density_2d_normalized_re(self, R_pc):
        re_pc = self.params.re_pc
        return 4.0 / (1.0 + (R_pc / re_pc) ** 2) ** 2

    def density_3d(self, r_pc):
        re_pc = self.params.re_pc
        return (3.0 / (4.0 * np.pi * re_pc**3)) / np.sqrt(
            1.0 + (r_pc / re_pc) ** 2
        ) ** 5

    def cdf_R(self, R_pc):
        r"""Return :math:`\int_0^R 2\pi R'\Sigma(R')\,dR'`."""
        re_pc = self.params.re_pc
        return 1.0 / (1.0 + (re_pc / R_pc) ** 2)

    def mean_density_2d(self, R_pc):
        re_pc = self.params.re_pc
        return 1.0 / np.pi / (R_pc**2 + re_pc**2)

    def _half_light_radius(self, re_pc):
        return re_pc

    def half_light_radius(self):
        return self._half_light_radius(self.params.re_pc)


class Exp2dModel(StellarModel):
    """Stellar model with an exponential projected surface density."""

    name = "Exp2dModel"
    required_param_names = ["re_pc"]
    required_models = {}

    @property
    def R_exp_pc(self):
        return self.params.re_pc / 1.67834699001666

    def density_2d(self, R_pc):
        scale = self.R_exp_pc
        return np.exp(-R_pc / scale) / (2.0 * np.pi * scale**2)

    def logdensity_2d(self, R_pc):
        scale = self.R_exp_pc
        return np.log(1.0 / (2.0 * np.pi)) - 2.0 * np.log(scale) - R_pc / scale

    def density_3d(self, r_pc):
        scale = self.R_exp_pc
        return k0(r_pc / scale) / (2.0 * np.pi**2 * scale**3)

    def cdf_R(self, R_pc):
        scale = self.R_exp_pc
        return 1.0 - np.exp(-R_pc / scale) * (1.0 + R_pc / scale)

    def mean_density_2d(self, R_pc):
        return self.cdf_R(R_pc) / (np.pi * R_pc**2)

    def _half_light_radius(self, re_pc):
        del re_pc
        return 1.67834699001666 * self.R_exp_pc

    def half_light_radius(self):
        return self._half_light_radius(self.params.re_pc)


class Exp3dModel(StellarModel):
    """Historical exponential model retained by the classical backend."""

    name = "Exp3dModel"
    required_param_names = ["re_pc"]
    required_models = {}

    def density_2d(self, R_pc):
        re_pc = self.params.re_pc
        return np.exp(-R_pc / re_pc) / (2.0 * np.pi * re_pc**2)

    def density_3d(self, r_pc):
        re_pc = self.params.re_pc
        return k0(r_pc / re_pc) / (2.0 * np.pi**2 * re_pc**3)

    def cdf_R(self, R_pc):
        re_pc = self.params.re_pc
        return 1.0 - np.exp(-R_pc / re_pc) * (1.0 + R_pc / re_pc)

    def mean_density_2d(self, R_pc):
        return self.cdf_R(R_pc) / (np.pi * R_pc**2)

    def half_light_radius(self):
        return 1.67834699001666 * self.params.re_pc


class Uniform2dModel(StellarModel):
    name = "uniform Model"
    required_param_names = ["Rmax_pc"]
    required_models = {}

    def density_2d(self, R_pc):
        return np.ones_like(R_pc) / (np.pi * self.params.Rmax_pc**2)

    def density_3d(self, r_pc):
        raise NotImplementedError("Uniform2dModel has no 3-D density model.")

    def cdf_R(self, R_pc):
        return (R_pc / self.params.Rmax_pc) ** 2


class DMModel(Model):
    """Base class for classical dark-matter density profiles."""

    name = "DM Model"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi_deg_max_warning = 1.0

    @abstractmethod
    def mass_density_3d(self, r_pc):
        raise NotImplementedError

    def enclosed_mass(self, r_pc):
        """Return mass enclosed within ``r_pc``.

        ``enclosure_mass`` is retained on concrete models for compatibility
        with the original API.
        """
        return self.enclosure_mass(r_pc)

    def _validate_jfactor_inputs(
        self,
        dist_pc,
        roi_deg,
        *,
        full=False,
        small_angle=False,
    ):
        try:
            dist_pc = np.asarray(dist_pc, dtype=float)
            roi_deg = np.asarray(roi_deg, dtype=float)
            r_t_pc = np.asarray(self.params["r_t_pc"], dtype=float)
            dist_pc, roi_deg, r_t_pc = np.broadcast_arrays(
                dist_pc, roi_deg, r_t_pc
            )
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
            raise ValueError(
                "dist_pc, roi_deg, and r_t_pc must be finite and positive."
            )

        if small_angle and np.any(roi_deg > self.roi_deg_max_warning):
            raise ValueError(
                "Small-angle J-factor approximations require roi_deg <= "
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

    def assert_roi_is_enough_small(self, roi_deg):
        try:
            roi_deg = np.asarray(roi_deg, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("roi_deg must be a finite positive number.") from exc
        if np.any(~np.isfinite(roi_deg)) or np.any(roi_deg <= 0):
            raise ValueError("roi_deg must be a finite positive number.")
        if np.any(roi_deg > self.roi_deg_max_warning):
            raise ValueError(
                "Small-angle J-factor approximations require roi_deg <= "
                f"{self.roi_deg_max_warning} degrees."
            )

    def jfactor_ullio2016_simple(self, dist_pc, roi_deg=0.5):
        """Calculate the spherical-aperture approximation to the J-factor."""
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
            rho = float(np.asarray(self.mass_density_3d(r_pc)))
            return r_pc**2 * rho**2

        integ, _ = quad(
            integrand,
            0.0,
            r_max_pc,
            epsabs=0.0,
            epsrel=1.0e-8,
            limit=300,
        )
        return C_J * 4.0 * np.pi / dist_pc**2 * integ

    def jfactor_ullio2016(self, dist_pc, roi_deg=0.5):
        """Calculate the full finite-ROI Ullio & Valli (2016) J-factor."""
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
    r"""General Zhao profile.

    .. math::

        \rho(r)=\rho_s (r/r_s)^{-g}
        [1+(r/r_s)^a]^{-(b-g)/a}.
    """

    name = "Zhao Model"
    required_param_names = ["rs_pc", "rhos_Msunpc3", "a", "b", "g", "r_t_pc"]
    required_models = {}

    def mass_density_3d(self, r_pc):
        rs_pc = self.params.rs_pc
        rhos = self.params.rhos_Msunpc3
        a, b, g = self.params.a, self.params.b, self.params.g
        x = np.asarray(r_pc) / rs_pc
        return rhos * np.power(x, -g) * np.power(
            1.0 + np.power(x, a), -(b - g) / a
        )

    def enclosure_mass(self, r_pc):
        rs_pc = self.params.rs_pc
        rhos = self.params.rhos_Msunpc3
        a, b, g = self.params.a, self.params.b, self.params.g
        r_t_pc = self.params.r_t_pc

        r_pc_trunc = np.minimum(np.asarray(r_pc), r_t_pc)

        if (
            np.isclose(a, 1.0, atol=1e-7, rtol=0.0)
            and np.isclose(b, 3.0, atol=1e-7, rtol=0.0)
            and np.isclose(g, 1.0, atol=1e-7, rtol=0.0)
        ):
            x_nfw = r_pc_trunc / rs_pc
            return (
                4.0
                * np.pi
                * rs_pc**3
                * rhos
                * (np.log1p(x_nfw) - x_nfw / (1.0 + x_nfw))
            )

        x = np.power(r_pc_trunc / rs_pc, a)
        argbeta0 = (3.0 - g) / a
        argbeta1 = (b - 3.0) / a
        return (
            4.0
            * np.pi
            * rs_pc**3
            * rhos
            / a
            * beta(argbeta0, argbeta1)
            * betainc(argbeta0, argbeta1, x / (1.0 + x))
        )


class NFWModel(DMModel):
    name = "NFW Model"
    required_param_names = ["rs_pc", "rhos_Msunpc3", "r_t_pc"]
    required_models = {}

    def mass_density_3d(self, r_pc):
        rs_pc = self.params.rs_pc
        rhos = self.params.rhos_Msunpc3
        x = np.asarray(r_pc) / rs_pc
        return rhos / x / (1.0 + x) ** 2

    def enclosure_mass(self, r_pc):
        threshold = 1e-7
        rs_pc = self.params.rs_pc
        rhos = self.params.rhos_Msunpc3
        r_t_pc = self.params.r_t_pc
        r_pc_trunc = np.minimum(np.asarray(r_pc), r_t_pc)
        x = r_pc_trunc / rs_pc
        value = np.log1p(x) - x / (1.0 + x)
        value = np.where(x < threshold, np.asarray(x) ** 2 / 2.0, value)
        return 4.0 * np.pi * rs_pc**3 * rhos * value

    def jfactor_ullio2016_simple(self, dist_pc, roi_deg=0.5):
        """Calculate the NFW spherical-aperture J-factor approximation."""
        dist_pc, roi_deg, r_t_pc = self._validate_jfactor_inputs(
            dist_pc, roi_deg, small_angle=True
        )
        roi_pc = dist_pc * np.sin(np.deg2rad(roi_deg))
        rs_pc = self.params.rs_pc
        rhos = self.params.rhos_Msunpc3
        r_max_pc = np.minimum(roi_pc, r_t_pc)
        c_max = r_max_pc / rs_pc
        j = C_J * 4.0 * np.pi * rs_pc**3 * rhos**2 / dist_pc**2
        j *= (
            (1.0 - 1.0 / (1.0 + c_max) ** 3) / 3.0
            + (rs_pc / dist_pc) ** 2
            * c_max**3
            / (1.0 + c_max) ** 3
            / 9.0
        )
        return j

    def jfactor_evans2016(self, dist_pc, roi_deg=0.5):
        """Evaluate the Evans et al. (2016) NFW J-factor fitting formula."""
        self.assert_roi_is_enough_small(roi_deg)

        def func_x(s):
            epsilon = 1e-8
            s = np.atleast_1d(s)
            if np.any(s < 0):
                raise ValueError("The Evans J-factor variable must be non-negative.")
            ret = np.full_like(s, np.nan, dtype=float)
            cond_1 = s < 1.0 - epsilon
            cond_2 = s > 1.0 + epsilon
            cond_3 = np.abs(1.0 - s) <= epsilon
            ret[cond_1] = np.arccosh(1.0 / s[cond_1]) / np.sqrt(
                1.0 - s[cond_1] ** 2
            )
            ret[cond_2] = np.arccos(1.0 / s[cond_2]) / np.sqrt(
                s[cond_2] ** 2 - 1.0
            )
            ret[cond_3] = (
                1.0
                - 2.0 * (s[cond_3] - 1.0) / 3.0
                + 7.0 * (s[cond_3] - 1.0) ** 2 / 15.0
            )
            return ret

        roi_pc = dist_pc * np.deg2rad(roi_deg)
        rs_pc = self.params.rs_pc
        rhos = self.params.rhos_Msunpc3
        r_t_pc = self.params.r_t_pc
        r_max_pc = np.minimum(roi_pc, r_t_pc)
        y = np.atleast_1d(np.asarray(r_max_pc / rs_pc, dtype=float))
        delta = 1.0 - y**2
        coeff_evans = (
            2.0 * y * (7.0 * y - 4.0 * y**3 + 3.0 * np.pi * delta**2)
            + 6.0
            * (2.0 * delta**3 - 2.0 * delta - y**4)
            * func_x(y)
        ) / (6.0 * delta**2)
        near_one = np.abs(1.0 - y) < 1e-8
        coeff_evans[near_one] = (
            np.pi
            - 38.0 / 15.0
            + (-64.0 / 21.0 + np.pi) * (y[near_one] - 1.0)
        )
        result = C_J * 2.0 * np.pi * rhos**2 * rs_pc**3 / dist_pc**2 * coeff_evans
        return result[0] if np.ndim(r_max_pc) == 0 else result


class AnisotropyModel(Model):
    name = "AnisotropyModel"

    @abstractmethod
    def beta(self, r):
        raise NotImplementedError

    @abstractmethod
    def f(self, r):
        raise NotImplementedError

    @abstractmethod
    def kernel(self, u, R, **kwargs):
        raise NotImplementedError


class ConstantAnisotropyModel(AnisotropyModel):
    name = "ConstantAnisotropyModel"
    required_param_names = ["beta_ani"]
    required_models = {}

    def beta(self, r):
        del r
        return self.params.beta_ani

    def f(self, r):
        return r ** (2.0 * self.params.beta_ani)

    def kernel(self, u, R, **kwargs):
        del R, kwargs
        b = self.params.beta_ani
        u2 = u**2
        return np.sqrt(1.0 - 1.0 / u2) * (
            (1.5 - b) * u2 * hyp2f1(1.0, 1.5 - b, 1.5, 1.0 - u2) - 0.5
        )


class OsipkovMerrittModel(AnisotropyModel):
    name = "OsipkovMerrittModel"
    required_param_names = ["r_a"]
    required_models = {}

    def beta(self, r):
        r_a = self.params.r_a
        return r**2 / (r**2 + r_a**2)

    def f(self, r):
        r_a = self.params.r_a
        return (r_a**2 + r**2) / r_a**2

    def kernel(self, u, R, **kwargs):
        del kwargs
        u_a = self.params.r_a / R
        u2_a = u_a**2
        u2 = u**2
        return (
            (u2 + u2_a)
            * (u2_a + 0.5)
            / (u * (u2_a + 1.0) ** 1.5)
            * np.arctan(np.sqrt((u2 - 1.0) / (u2_a + 1.0)))
            - np.sqrt(1.0 - 1.0 / u2) / (2.0 * (u2_a + 1.0))
        )


class BaesAnisotropyModel(AnisotropyModel):
    name = "BaesAnisotropyModel"
    required_param_names = ["beta_0", "beta_inf", "r_a", "eta"]
    required_models = {}

    def beta(self, r):
        b0, binf = self.params.beta_0, self.params.beta_inf
        r_a, eta = self.params.r_a, self.params.eta
        x = np.power(r / r_a, eta)
        return (b0 + binf * x) / (1.0 + x)

    def f(self, r):
        b0, binf = self.params.beta_0, self.params.beta_inf
        r_a, eta = self.params.r_a, self.params.eta
        x = np.power(r / r_a, eta)
        return np.power(r, 2.0 * b0) * np.power(
            1.0 + x, 2.0 * (binf - b0) / eta
        )

    def integrand_kernel(self, u_integ, R):
        u2_integ = u_integ**2
        r_integ = R * u_integ
        return (
            u_integ
            / np.sqrt(u2_integ - 1.0)
            * (1.0 - self.beta(r_integ) / u2_integ)
            / self.f(r_integ)
        )

    def kernel(self, u, R, **kwargs):
        n = kwargs.get("n", 128)
        u = np.asarray(u).reshape(-1)
        R = np.asarray(R).reshape(-1)

        u_expanded = u[np.newaxis, :, np.newaxis]
        R_expanded = R[:, np.newaxis, np.newaxis]

        def integrand(_u):
            return self.integrand_kernel(_u, R_expanded)

        integration = dequad(
            integrand,
            1,
            u_expanded,
            n,
            axis=2,
            replace_inf_to_zero=True,
            replace_nan_to_zero=True,
        )
        return (
            integration
            * self.f(R_expanded[..., 0] * u_expanded[..., 0])
            / u_expanded[..., 0]
        )


__all__ = [
    "AnisotropyModel",
    "BaesAnisotropyModel",
    "ConstantAnisotropyModel",
    "DMModel",
    "Exp2dModel",
    "Exp3dModel",
    "NFWModel",
    "OsipkovMerrittModel",
    "PlummerModel",
    "StellarModel",
    "Uniform2dModel",
    "ZhaoModel",
]
