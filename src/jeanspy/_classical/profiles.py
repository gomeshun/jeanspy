"""Physical profile components for the classical NumPy/SciPy backend."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1, k0

from ..dequad import dequad
from .._zhao import enclosed_mass as _zhao_mass, valid_domain as _zhao_valid
from .core import Model
from .jfactor import C_J, _ullio2016_inner_weight, _ullio2016_weight


def _jfactor_quad(integrand, lo, hi, **kwargs):
    """Do not return an unconverged or nonphysical quadrature as a J-factor."""
    result = quad(integrand, lo, hi, full_output=True, **kwargs)
    value, error = result[:2]
    if len(result) != 3 or not np.isfinite(value) or value < 0:
        detail = result[3] if len(result) > 3 else "nonfinite or negative integral"
        raise ValueError(f"J-factor quadrature failed: {detail}")
    return value, error


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
        R = np.asarray(R_pc, dtype=float)
        cutoff = np.asarray(R_trunc_pc, dtype=float)
        if cutoff.ndim != 0 or not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("R_trunc_pc must be a finite positive scalar")
        if np.any(~np.isfinite(R)) or np.any(R < 0):
            raise ValueError("R_pc must be finite and nonnegative")
        norm = self.cdf_R(cutoff)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Truncated stellar density must have positive finite normalization")
        return np.where(R <= cutoff, self.density_2d(R) / norm, 0.0)

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
        R, rmax = self._validated_radii(R_pc)
        return np.where((R >= 0) & (R <= rmax), 1.0 / (np.pi * rmax**2), 0.0)

    def density_3d(self, r_pc):
        raise NotImplementedError("Uniform2dModel has no 3-D density model.")

    def cdf_R(self, R_pc):
        R, rmax = self._validated_radii(R_pc)
        return (np.clip(R, 0.0, rmax) / rmax) ** 2

    def _validated_radii(self, R_pc):
        R = np.asarray(R_pc, dtype=float)
        rmax = np.asarray(self.params.Rmax_pc, dtype=float)
        if rmax.ndim != 0 or not np.isfinite(rmax) or rmax <= 0:
            raise ValueError("Rmax_pc must be a finite positive scalar")
        if np.any(np.isnan(R)):
            raise ValueError("R_pc must not contain NaN")
        return R, rmax


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

        self._validate_jfactor_profile()
        return dist_pc, roi_deg, r_t_pc

    def _validate_jfactor_profile(self):
        """Subclasses must check any analytic convergence restrictions."""
        for name in self.required_param_names:
            if np.any(~np.isfinite(np.asarray(self.params[name], dtype=float))):
                raise ValueError(f"J-factor profile parameter {name} must be finite")

    def _jfactor_density(self, r_pc):
        rho = float(np.asarray(self.mass_density_3d(r_pc)))
        if not np.isfinite(rho) or rho < 0:
            raise ValueError("J-factor density must be finite and nonnegative away from the origin")
        return rho

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
        """Calculate the small-angle spherical-aperture approximation.

        Let ``R_max = dist_pc * sin(roi_deg)``.  This method integrates the
        spherical luminosity only out to ``min(R_max, r_t_pc)``.

        If ``R_max >= r_t_pc``, the aperture contains the entire truncated
        halo and this reduces to Ullio & Valli (2016), Eq. (B.10), with the
        halo boundary ``mathcal R = r_t_pc``.  If ``R_max < r_t_pc``, this is
        instead a spherical-aperture approximation: projected contributions
        from shells with ``R_max < r < r_t_pc`` are omitted.  Use
        :meth:`jfactor_ullio2016` for the full finite-ROI geometry of
        Eqs. (B.8)--(B.9).
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
            rho = self._jfactor_density(r_pc)
            return r_pc**2 * rho**2

        integ, _ = _jfactor_quad(
            integrand,
            0.0,
            r_max_pc,
            epsabs=0.0,
            epsrel=1.0e-8,
            limit=300,
        )
        return C_J * 4.0 * np.pi / dist_pc**2 * integ

    def jfactor_ullio2016(self, dist_pc, roi_deg=0.5):
        """Calculate the full finite-ROI Ullio & Valli (2016) J-factor.

        Unlike :meth:`jfactor_ullio2016_simple`, this includes the projected
        contribution from shells with ``R_max < r < r_t_pc`` when the ROI is
        smaller than the truncated halo, following Eqs. (B.8)--(B.9).
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
            rho = self._jfactor_density(r_pc)
            return rho**2 * float(_ullio2016_inner_weight(r_pc, dist_pc))

        integ, _ = _jfactor_quad(
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
                rho = self._jfactor_density(r_pc)
                weight = _ullio2016_weight(r_pc, 0.0, r_max_pc, dist_pc)
                return 2.0 * outer_width_pc * u * rho**2 * float(weight)

            outer, _ = _jfactor_quad(
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

    def _validate_jfactor_profile(self):
        super()._validate_jfactor_profile()
        if (not np.all(_zhao_valid(np.asarray(1.0), self.params, np))
                or np.any(np.asarray(self.params.g) >= 1.5)):
            raise ValueError("Finite Zhao J-factor requires positive scales, a > 0, and g < 1.5; "
                             "steeper central cusps have divergent annihilation luminosity")

    def mass_density_3d(self, r_pc):
        rs_pc = self.params.rs_pc
        rhos = self.params.rhos_Msunpc3
        a, b, g = self.params.a, self.params.b, self.params.g
        x = np.asarray(r_pc) / rs_pc
        return rhos * np.power(x, -g) * np.power(
            1.0 + np.power(x, a), -(b - g) / a
        )

    def enclosed_mass(self, r_pc, *, n_steps=128):
        """Finite-radius Zhao mass for a > 0 and g < 3, including b <= 3.

        n_steps controls Gauss-Legendre nodes per regularized segment.
        """
        params = {k: getattr(self.params, k) for k in self.required_param_names}
        if not np.all(_zhao_valid(np.asarray(r_pc), params, np)):
            raise ValueError(
                "Invalid Zhao mass domain: require positive scales, a > 0, "
                "g < 3, finite slopes and nonnegative truncated radii"
            )
        return _zhao_mass(r_pc, params, xp=np, n_steps=n_steps)

    def enclosure_mass(self, r_pc, *, n_steps=128):
        return self.enclosed_mass(r_pc, n_steps=n_steps)


class NFWModel(DMModel):
    name = "NFW Model"
    required_param_names = ["rs_pc", "rhos_Msunpc3", "r_t_pc"]
    required_models = {}

    def _validate_jfactor_profile(self):
        super()._validate_jfactor_profile()
        if np.any(np.asarray(self.params.rs_pc) <= 0) or np.any(np.asarray(self.params.rhos_Msunpc3) <= 0):
            raise ValueError("NFW J-factor requires positive rs_pc and rhos_Msunpc3")

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
        """Evaluate the NFW spherical-aperture approximation analytically.

        The geometric interpretation is the same as
        :meth:`DMModel.jfactor_ullio2016_simple`: for ``R_max >= r_t_pc`` the
        aperture encloses the full truncated halo and corresponds to the
        Eq. (B.10) limit; for ``R_max < r_t_pc`` it omits projected outer-shell
        contributions.  Use :meth:`jfactor_ullio2016` for the full finite-ROI
        geometry.
        """
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
        """Evaluate the small-angle, infinite-LOS Evans et al. (2016) formula.

        This historical approximation caps the *projected aperture* at r_t_pc;
        it does not truncate the density along the line of sight. For a halo
        truncated in three dimensions use jfactor_ullio2016 instead.
        """
        dist_pc, roi_deg, r_t_pc = self._validate_jfactor_inputs(
            dist_pc, roi_deg, small_angle=True
        )
        r_max_pc = np.minimum(dist_pc * np.deg2rad(roi_deg), r_t_pc)
        rs_pc, rhos = self.params.rs_pc, self.params.rhos_Msunpc3
        y = np.asarray(r_max_pc / rs_pc, dtype=float)
        shape = y.shape
        y = y.reshape(-1)
        delta = (1.0 - y) * (1.0 + y)
        coeff = np.empty_like(y)
        near = np.abs(delta) <= 0.2
        # X(y) = sum(delta**n / (2*n+1)); cancel the constant and linear
        # terms symbolically before dividing the Evans numerator by delta**2.
        # Twenty terms give an absolute remainder < 1e-16 on |delta| <= 0.2.
        series = [-38.0 / 15.0] + [
            2.0 / (2*n - 1) - 1.0 / (2*n + 1) - 1.0 / (2*n + 5)
            for n in range(1, 21)
        ]
        coeff[near] = np.pi * y[near] + np.polynomial.polynomial.polyval(delta[near], series)
        far = ~near
        s, d = y[far], delta[far]
        x = np.empty_like(s)
        below = s < 1.0
        x[below] = np.arccosh(1.0 / s[below]) / np.sqrt(d[below])
        x[~below] = np.arccos(1.0 / s[~below]) / np.sqrt(-d[~below])
        coeff[far] = (2*s*(7*s - 4*s**3 + 3*np.pi*d**2)
                      + 6*(2*d**3 - 2*d - s**4)*x) / (6*d**2)
        result = C_J * 2*np.pi*rhos**2*rs_pc**3 / dist_pc**2 * coeff.reshape(shape)
        if np.any(~np.isfinite(result)) or np.any(result <= 0):
            raise ValueError("Evans J-factor evaluation is nonfinite or nonpositive")
        return result.item() if result.ndim == 0 else result


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
