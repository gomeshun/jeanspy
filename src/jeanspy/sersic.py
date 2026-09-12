"""Spherical Sérsic profile for the public classical JeansPy model API."""

from __future__ import annotations

from importlib.resources import files
from typing import Optional

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import gamma, gammainc

from ._classical.profiles import StellarModel
from ._sersic_deprojection import sp04_density


class SersicModel(StellarModel):
    r"""Projected Sérsic stellar model with selectable 3-D deprojection.

    The numerical Abel inversion is the reference implementation. Fast
    approximations remain explicitly selectable, while ``"auto"`` uses the
    Vitral & Mamon (2021) hybrid where supported and falls back to the numerical
    reference outside the approximation domain.

    Notes
    -----
    **Inputs and units.** ``re_pc`` is projected half-light radius (pc); n is
    dimensionless Sersic index; ``deprojection_method`` is
    auto/approx/vm20/vm20bis/numerical. Other constructor arguments follow
    Model.

    **Returns and shape.** ``density_2d`` and ``density_3d`` return pc^-2 and
    pc^-3 with input shape. ``cdf_R`` returns the dimensionless projected radial
    CDF; ``half_light_radius`` returns pc. ``mean_density_2d`` is mean surface
    density within R. ``logdensity_2d`` is the natural log of the surface
    density, not the radial PDF. ``density_2d_normalized_re``, where available,
    is the dimensionless ratio Sigma(R)/Sigma(re).

    **Validity.** Supply positive finite scales and real nonnegative radii. Use
    NumPy arrays for array inputs. A central cusp may diverge at zero. The older
    elementary density formulas do not uniformly validate domains. The bundled
    ``b_n`` interpolator covers n from about 0.02 to 15.17; numerical
    deprojection does not remove that table limit. VM20 and VM20bis have the
    stricter n/r domains specified by their methods.

    **Errors.** Unknown parameter names raise ValueError in Model.update.
    Invalid values in elementary profile formulas can produce NaN/inf;
    successful construction alone does not validate a physical profile.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_profiles.py``
    """

    name = "SersicModel"
    required_param_names = ["re_pc", "n"]
    required_models = {}

    _VALID_DEPROJECTION_METHODS = (
        "auto",
        "approx",
        "vm20",
        "vm20bis",
        "numerical",
    )

    def __init__(self, *args, deprojection_method: str = "auto", **kwargs):
        if deprojection_method not in self._VALID_DEPROJECTION_METHODS:
            raise ValueError(
                "deprojection_method must be one of "
                f"{self._VALID_DEPROJECTION_METHODS!r}, got {deprojection_method!r}"
            )

        super().__init__(*args, **kwargs)

        data_dir = files("jeanspy").joinpath("data")
        bn_table = np.genfromtxt(
            data_dir.joinpath("sersic_log10n_log10bn.csv"),
            delimiter=",",
            names=True,
        )
        self._b_interp = interp1d(
            bn_table["log10n"],
            bn_table["log10bn"],
            kind="cubic",
            assume_sorted=True,
        )
        self.coeff = np.loadtxt(
            data_dir.joinpath("coeff_dens.csv"),
            comments="#",
        )
        self.coeff_vm20bis = np.loadtxt(
            data_dir.joinpath("coeff_dens_vm20bis.csv"),
            comments="#",
        )
        self.deprojection_method = deprojection_method

    @property
    def b_approx(self):
        """Return the historical approximation b_n = 2*n - 0.324.

        This helper does not impose a validity interval and is not the tabulated
        b_n used by the normalized surface-density formula.
        """
        return 2.0 * self.params.n - 0.324

    @property
    def b_CB(self):
        """Ciotti & Bertin (1999) approximation to the Sérsic ``b_n``."""
        n = self.params.n
        return (
            2.0 * n
            - 1.0 / 3.0
            + 4.0 / (405.0 * n)
            + 46.0 / (25515.0 * n**2)
            + 131.0 / (1148175.0 * n**3)
            - 2194697.0 / (30690717750.0 * n**4)
        )

    @property
    def b(self):
        """Return the dimensionless Sersic b_n from the bundled interpolation table.

        The supported n domain is the table range stated in the class contract.
        """
        return 10 ** self._b_interp(np.log10(self.params.n))

    @property
    def norm(self):
        """Return the projected Sersic normalization in pc^2 for stored n and re_pc."""
        n = self.params.n
        return (
            np.pi
            * self.params.re_pc**2
            * np.power(self.b, -2.0 * n)
            * gamma(2.0 * n + 1.0)
        )

    def density_2d(self, R_pc):
        r"""Evaluate normalized projected tracer density.

        Notes
        -----
        **Inputs and units.** ``R_pc`` is a scalar or NumPy array of projected radii
        in pc.

        **Returns and shape.** pc^-2 with input radius shape.
        """
        n = self.params.n
        return np.exp(
            -self.b * np.power(np.asarray(R_pc) / self.params.re_pc, 1.0 / n)
        ) / self.norm

    def density_2d_normalized_re(self, R_pc):
        """Return the dimensionless projected density ratio Sigma(R)/Sigma(re).

        R_pc is a scalar or NumPy array in pc; output follows its shape.
        The stored re_pc and n must lie in the supported positive domain.
        """
        n = self.params.n
        return np.exp(
            -self.b
            * (
                np.power(np.asarray(R_pc) / self.params.re_pc, 1.0 / n)
                - 1.0
            )
        )

    def cdf_R(self, R_pc):
        r"""Return :math:`\int_0^R 2\pi R'\Sigma(R')\,dR'`.

        Notes
        -----
        **Inputs and units.** ``R_pc`` is a scalar or NumPy array of projected radii
        in pc.

        **Returns and shape.** Dimensionless cumulative probability with input
        radius shape.
        """
        n = self.params.n
        return gammainc(
            2.0 * n,
            self.b * np.power(np.asarray(R_pc) / self.params.re_pc, 1.0 / n),
        )

    def mean_density_2d(self, R_pc):
        r"""Average surface density inside a circular aperture.

        Notes
        -----
        **Inputs and units.** Positive projected radius ``R_pc`` in pc, scalar or
        NumPy array.

        **Returns and shape.** ``cdf_R(R)/(pi*R**2)``, in pc^-2 with radius shape.

        **Validity.** Use R>0; the elementary ratio is not a numerically regularized
        central limit.
        """
        return self.cdf_R(R_pc) / (np.pi * np.asarray(R_pc) ** 2)

    @property
    def p_LGM(self):
        """Return the dimensionless LGM approximate deprojection exponent for stored n."""
        n = self.params.n
        return 1.0 - 0.6097 / n + 0.05463 / n**2

    @property
    def norm_3d(self):
        """Return the LGM approximate deprojection normalization in pc^3.

        Uses the stored n, re_pc, b_CB and p_LGM. This is the LGM approximation
        normalization, not the numerical Abel-deprojection normalization.
        """
        re = self.params.re_pc
        n = self.params.n
        b = self.b_CB
        p = self.p_LGM
        index = (3.0 - p) * n
        return 4.0 * np.pi * re**3 * n * gamma(index) / b**index

    def density_3d_LGM(self, r_pc):
        r"""Legacy Lima Neto--Gerbal--Márquez Sérsic deprojection.

        Notes
        -----
        **Inputs and units.** ``r_pc`` is scalar or an array in pc; n and ``re_pc``
        come from stored parameters.

        **Returns and shape.** Unit-integral tracer density in pc^-3 with input
        shape.

        **Validity.** Approximate deprojection; it is not a general reference or
        central-limit formula.
        """
        n = float(self.params.n)
        if not (0.5 <= n <= 10.0):
            raise ValueError(
                f"density_3d_LGM is supported for 0.5 ≤ n ≤ 10; got n={n}."
            )
        p = self.p_LGM
        b = self.b_CB
        x = np.asarray(r_pc) / self.params.re_pc
        return x ** (-p) * np.exp(-b * x ** (1.0 / n)) / self.norm_3d

    def half_light_radius(self):
        r"""Return the projected half-light radius.

        Notes
        -----
        **Inputs and units.** No arguments; reads the stored tracer scales.

        **Returns and shape.** Scalar radius in pc. Exp3dModel returns
        1.67834699001666\*``re_pc``.
        """
        return self.params.re_pc

    @staticmethod
    def _eval_vm20_poly(coeff_table, log_x, log_n):
        """Evaluate a VM20-family logarithmic density correction."""
        p = 0.0
        order = coeff_table.shape[0] - 1
        for l in range(order + 1):
            for j in range(order + 1 - l):
                p += coeff_table[l, j] * log_n**j * log_x**l
        return p

    def density_3d_VM20(self, r_pc):
        r"""Vitral & Mamon (2020) 3-D Sérsic density approximation.

        Notes
        -----
        **Inputs and units.** ``r_pc`` is scalar or an array in pc; n and ``re_pc``
        come from stored parameters.

        **Returns and shape.** Unit-integral tracer density in pc^-3 with input
        shape.

        **Validity.** 0.5<=n<=10; 1e-3<=r/re<=1e3; positive finite r.
        """
        n = float(self.params.n)
        re = float(self.params.re_pc)
        if not (0.5 <= n <= 10.0):
            raise ValueError(
                f"density_3d_VM20 is valid for 0.5 ≤ n ≤ 10; got n={n}."
            )

        scalar_input = np.ndim(r_pc) == 0
        r_arr = np.atleast_1d(np.asarray(r_pc, dtype=float))
        x_arr = r_arr / re
        if np.any(~np.isfinite(x_arr)) or np.any(x_arr <= 0.0):
            raise ValueError("r_pc must be finite and positive for density_3d_VM20.")

        log_x_arr = np.log10(x_arr)
        if np.any((log_x_arr < -3.0) | (log_x_arr > 3.0)):
            bad = np.flatnonzero(
                (log_x_arr.ravel() < -3.0) | (log_x_arr.ravel() > 3.0)
            )[0]
            x_bad = x_arr.ravel()[bad]
            raise ValueError(
                "density_3d_VM20 is valid for 1e-3 ≤ r/R_e ≤ 1e3; "
                f"got r/R_e = {x_bad:.3g}."
            )

        log_n = np.log10(n)
        result = np.empty_like(r_arr)
        for idx in np.ndindex(r_arr.shape):
            p = self._eval_vm20_poly(self.coeff, log_x_arr[idx], log_n)
            result[idx] = self.density_3d_LGM(r_arr[idx]) * 10**p

        if scalar_input:
            return float(result.ravel()[0])
        return result

    def density_3d_VM20bis(self, r_pc):
        r"""Official Vitral & Mamon (2021) VM20bis density approximation.

        Notes
        -----
        **Inputs and units.** ``r_pc`` is scalar or an array in pc; n and ``re_pc``
        come from stored parameters.

        **Returns and shape.** Unit-integral tracer density in pc^-3 with input
        shape.

        **Validity.** 0.5<=n<=3.4; 1e-4<=r/re<=1e3; positive finite r.
        """
        n = float(self.params.n)
        re = float(self.params.re_pc)
        if not (0.5 <= n <= 3.4):
            raise ValueError(
                f"density_3d_VM20bis is valid for 0.5 ≤ n ≤ 3.4; got n={n}."
            )

        scalar_input = np.ndim(r_pc) == 0
        r_arr = np.atleast_1d(np.asarray(r_pc, dtype=float))
        x_arr = r_arr / re
        if np.any(~np.isfinite(x_arr)) or np.any(x_arr <= 0.0):
            raise ValueError(
                "r_pc must be finite and positive for density_3d_VM20bis."
            )

        log_x_arr = np.log10(x_arr)
        if np.any((log_x_arr < -4.0) | (log_x_arr > 3.0)):
            bad = np.flatnonzero(
                (log_x_arr.ravel() < -4.0) | (log_x_arr.ravel() > 3.0)
            )[0]
            x_bad = x_arr.ravel()[bad]
            raise ValueError(
                "density_3d_VM20bis is valid for 1e-4 ≤ r/R_e ≤ 1e3; "
                f"got r/R_e = {x_bad:.3g}."
            )

        log_n = np.log10(n)
        result = np.empty_like(r_arr)
        for idx in np.ndindex(r_arr.shape):
            p = self._eval_vm20_poly(
                self.coeff_vm20bis,
                log_x_arr[idx],
                log_n,
            )
            result[idx] = self.density_3d_LGM(r_arr[idx]) * 10**p

        if scalar_input:
            return float(result.ravel()[0])
        return result

    def density_3d_numerical(
        self,
        r_pc,
        *,
        epsrel: float = 1e-6,
        epsabs: float = 0.0,
        limit: int = 200,
    ):
        r"""Deproject the Sérsic surface density by numerical Abel inversion.

        Notes
        -----
        **Inputs and units.** Nonnegative ``r_pc`` in pc; epsrel/epsabs are
        adaptive-integral tolerances; limit is the subdivision limit.

        **Returns and shape.** pc^-3 with radius shape; infinity at r=0 for n>=1, a
        finite analytic center for n<1, zero at +infinity.

        **Validity.** Positive finite ``re_pc`` and supported interpolation-table n.
        At r>0 integrate over theta in [0,pi/2].

        **Errors.** Negative/NaN radii raise ValueError; SciPy convergence warnings
        can propagate. The returned value is not accompanied by a certified
        deprojection error.
        """
        n = float(self.params.n)
        re = float(self.params.re_pc)
        b = float(self.b)
        norm2d = float(self.norm)

        def central_density():
            if n >= 1.0:
                return np.inf
            return (
                b ** (3.0 * n)
                * gamma(1.0 - n)
                / (2.0 * np.pi**2 * n * gamma(2.0 * n) * re**3)
            )

        def dsigma_dR(R):
            return (
                np.exp(-b * (R / re) ** (1.0 / n))
                / norm2d
                * (-b / n)
                * (R / re) ** (1.0 / n - 1.0)
                / re
            )

        def rho_scalar(r):
            if np.isnan(r):
                raise ValueError("r_pc must not be NaN.")
            if r < 0:
                raise ValueError(f"r_pc must be non-negative; got {r}.")
            if r == 0.0:
                return central_density()
            if np.isposinf(r):
                return 0.0

            def integrand(theta):
                cos_theta = np.cos(theta)
                R = r / cos_theta
                return dsigma_dR(R) / cos_theta

            value, _ = quad(
                integrand,
                0.0,
                np.pi / 2.0,
                limit=limit,
                epsrel=epsrel,
                epsabs=epsabs,
            )
            return -value / np.pi

        scalar_input = np.ndim(r_pc) == 0
        r_arr = np.atleast_1d(np.asarray(r_pc, dtype=float))
        result = np.array([rho_scalar(r) for r in r_arr.ravel()], dtype=float)
        if scalar_input:
            return float(result[0])
        return result.reshape(r_arr.shape)

    def density_3d_auto(self, r_pc):
        r"""Safely choose a fast literature approximation or numerical Abel.

        Notes
        -----
        **Inputs and units.** ``r_pc`` is scalar or an array in pc; n and ``re_pc``
        come from stored parameters.

        **Returns and shape.** Unit-integral tracer density in pc^-3 with input
        shape.

        **Validity.** Uses VM20bis, SP04 or numerical Abel inversion according to
        the documented n/r domain.
        """
        n = float(self.params.n)
        re = float(self.params.re_pc)
        scalar_input = np.ndim(r_pc) == 0
        r_arr = np.atleast_1d(np.asarray(r_pc, dtype=float))

        if np.any(np.isnan(r_arr)):
            raise ValueError("r_pc must not contain NaN.")
        if np.any(r_arr < 0):
            raise ValueError("r_pc must be non-negative.")

        x_arr = r_arr / re
        approximation_mask = (
            (r_arr > 0.0)
            & np.isfinite(r_arr)
            & (x_arr >= 1e-4)
            & (x_arr <= 1e3)
            & (0.5 <= n)
            & (n <= 10.0)
        )

        result = np.empty_like(r_arr, dtype=float)
        numerical_mask = ~approximation_mask
        if np.any(numerical_mask):
            result[numerical_mask] = self.density_3d_numerical(
                r_arr[numerical_mask]
            )

        if np.any(approximation_mask):
            radii = r_arr[approximation_mask]
            if n <= 3.4:
                result[approximation_mask] = self.density_3d_VM20bis(radii)
            else:
                result[approximation_mask] = sp04_density(
                    radii,
                    re_pc=re,
                    n=n,
                    b=float(self.b),
                )

        if scalar_input:
            return float(result[0])
        return result.reshape(r_arr.shape)

    def density_3d(self, r_pc, method: Optional[str] = None):
        r"""Return the 3-D density using the requested deprojection method.

        Notes
        -----
        **Inputs and units.** ``r_pc`` is a scalar or NumPy array of intrinsic radii
        in pc. SersicModel also accepts method.

        **Returns and shape.** pc^-3 with input radius shape.
        """
        resolved = method if method is not None else self.deprojection_method
        if resolved not in self._VALID_DEPROJECTION_METHODS:
            raise ValueError(
                f"method must be one of {self._VALID_DEPROJECTION_METHODS!r}, "
                f"got {resolved!r}"
            )
        if resolved == "auto":
            return self.density_3d_auto(r_pc)
        if resolved == "approx":
            return self.density_3d_LGM(r_pc)
        if resolved == "vm20":
            return self.density_3d_VM20(r_pc)
        if resolved == "vm20bis":
            return self.density_3d_VM20bis(r_pc)
        return self.density_3d_numerical(r_pc)


__all__ = ["SersicModel"]
