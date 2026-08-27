"""Modern spherical Sérsic deprojection for the public JeansPy model API."""

from __future__ import annotations

from importlib.resources import files
from typing import Optional

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

from ._model_impl import SersicModel as _LegacySersicModel
from ._sersic_deprojection import sp04_density


class SersicModel(_LegacySersicModel):
    """Projected Sérsic stellar model with selectable 3-D deprojection.

    The numerical Abel inversion is the reference implementation. Fast
    approximations remain explicitly selectable, while ``"auto"`` uses the
    Vitral & Mamon (2021) hybrid where supported and falls back to the numerical
    reference outside the approximation domain.
    """

    _VALID_DEPROJECTION_METHODS = (
        "auto",
        "approx",
        "vm20",
        "vm20bis",
        "numerical",
    )

    def __init__(self, *args, deprojection_method: str = "auto", **kwargs):
        """Initialize the Sérsic model.

        Parameters
        ----------
        deprojection_method : {"auto", "approx", "vm20", "vm20bis", "numerical"}
            Method used by :meth:`density_3d` when ``method`` is omitted.

            ``"auto"`` (default)
                Use official VM20bis for ``0.5 <= n <= 3.4`` and SP04 for
                ``3.4 < n <= 10`` when ``1e-4 <= r/R_e <= 1e3``. At zero,
                outside this radius interval, or outside this Sérsic-index
                range, use numerical Abel inversion.
            ``"approx"``
                Legacy Lima Neto--Gerbal--Márquez approximation.
            ``"vm20"``
                Vitral & Mamon (2020), valid for ``0.5 <= n <= 10`` and
                ``1e-3 <= r/R_e <= 1e3``.
            ``"vm20bis"``
                Official Vitral & Mamon (2021) low-radius recalibration,
                valid for ``0.5 <= n <= 3.4`` and
                ``1e-4 <= r/R_e <= 1e3``.
            ``"numerical"``
                Numerical spherical Abel inversion (reference implementation).
        """
        if deprojection_method not in self._VALID_DEPROJECTION_METHODS:
            raise ValueError(
                "deprojection_method must be one of "
                f"{self._VALID_DEPROJECTION_METHODS!r}, got {deprojection_method!r}"
            )

        # The historical implementation provides common Sérsic initialization
        # (including the published VM20 coefficient table). Keep modern
        # deprojection policy entirely in this maintained subclass so that the
        # historical module can track the latest main-branch physics changes.
        super().__init__(*args, **kwargs)
        self.coeff_vm20bis = np.loadtxt(
            files("jeanspy").joinpath("data", "coeff_dens_vm20bis.csv"),
            comments="#",
        )
        self.deprojection_method = deprojection_method

    @staticmethod
    def _eval_vm20_poly(coeff_table, log_x, log_n):
        """Evaluate a VM20-family logarithmic density correction.

        The bundled coefficient tables use the convention

        ``rho_corrected = rho_LGM * 10**P``

        where ``P = log10(rho_corrected / rho_LGM)`` is the bivariate
        polynomial in ``log10(r/R_e)`` and ``log10(n)``.
        """
        p = 0.0
        order = coeff_table.shape[0] - 1
        for l in range(order + 1):
            for j in range(order + 1 - l):
                p += coeff_table[l, j] * log_n**j * log_x**l
        return p

    def density_3d_LGM(self, r_pc):
        """Legacy Lima Neto--Gerbal--Márquez Sérsic deprojection."""
        n = float(self.params.n)
        if not (0.5 <= n <= 10.0):
            raise ValueError(
                f"density_3d_LGM is supported for 0.5 ≤ n ≤ 10; got n={n}."
            )
        return super().density_3d_LGM(r_pc)

    def density_3d_VM20(self, r_pc):
        """Vitral & Mamon (2020) 3-D Sérsic density approximation.

        The correction uses

        ``rho_VM20 = rho_LGM * 10**P``,

        with ``P = log10(rho_VM20 / rho_LGM)``. It is calibrated for
        ``0.5 <= n <= 10`` and ``1e-3 <= r/R_e <= 1e3``; extrapolation is
        rejected.
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
            bad = np.flatnonzero((log_x_arr.ravel() < -3.0) | (log_x_arr.ravel() > 3.0))[0]
            x_bad = x_arr.ravel()[bad]
            raise ValueError(
                "density_3d_VM20 is valid for 1e-3 ≤ r/R_e ≤ 1e3; "
                f"got r/R_e = {x_bad:.3g}."
            )

        log_n = np.log10(n)
        result = np.empty_like(r_arr)
        for idx in np.ndindex(r_arr.shape):
            p = self._eval_vm20_poly(self.coeff, log_x_arr[idx], log_n)
            result[idx] = super().density_3d_LGM(r_arr[idx]) * 10**p

        if scalar_input:
            return float(result.ravel()[0])
        return result

    def density_3d_VM20bis(self, r_pc):
        """Official Vitral & Mamon (2021) VM20bis density approximation.

        Coefficients are taken from the authors' public ``coeff_dens.txt``
        table associated with Vitral & Mamon (2021). The polynomial uses
        ``rho_VM20bis = rho_LGM * 10**P`` with
        ``P = log10(rho_VM20bis / rho_LGM)`` and is supported for
        ``0.5 <= n <= 3.4`` and ``1e-4 <= r/R_e <= 1e3``.
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
            raise ValueError("r_pc must be finite and positive for density_3d_VM20bis.")

        log_x_arr = np.log10(x_arr)
        if np.any((log_x_arr < -4.0) | (log_x_arr > 3.0)):
            bad = np.flatnonzero((log_x_arr.ravel() < -4.0) | (log_x_arr.ravel() > 3.0))[0]
            x_bad = x_arr.ravel()[bad]
            raise ValueError(
                "density_3d_VM20bis is valid for 1e-4 ≤ r/R_e ≤ 1e3; "
                f"got r/R_e = {x_bad:.3g}."
            )

        log_n = np.log10(n)
        result = np.empty_like(r_arr)
        for idx in np.ndindex(r_arr.shape):
            p = self._eval_vm20_poly(self.coeff_vm20bis, log_x_arr[idx], log_n)
            result[idx] = super().density_3d_LGM(r_arr[idx]) * 10**p

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
        """Deproject the Sérsic surface density by numerical Abel inversion.

        The substitution ``R = r / cos(theta)`` removes the square-root endpoint
        singularity. At exactly ``r=0`` the known analytic central limit is
        used: it is finite for ``n < 1`` and divergent for ``n >= 1``.
        Positive infinity is mapped to the Sérsic asymptotic value zero.
        """
        n = float(self.params.n)
        re = float(self.params.re_pc)
        b = float(self.b)
        norm2d = float(self.norm)

        def _central_density():
            if n >= 1.0:
                return np.inf
            return (
                b ** (3.0 * n)
                * gamma(1.0 - n)
                / (2.0 * np.pi**2 * n * gamma(2.0 * n) * re**3)
            )

        def _dsigma_dR(R):
            return (
                np.exp(-b * (R / re) ** (1.0 / n))
                / norm2d
                * (-b / n)
                * (R / re) ** (1.0 / n - 1.0)
                / re
            )

        def _rho_scalar(r):
            if np.isnan(r):
                raise ValueError("r_pc must not be NaN.")
            if r < 0:
                raise ValueError(f"r_pc must be non-negative; got {r}.")
            if r == 0.0:
                return _central_density()
            if np.isposinf(r):
                return 0.0

            def integrand(theta):
                cos_theta = np.cos(theta)
                R = r / cos_theta
                return _dsigma_dR(R) / cos_theta

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
        result = np.array([_rho_scalar(r) for r in r_arr.ravel()], dtype=float)
        if scalar_input:
            return float(result[0])
        return result.reshape(r_arr.shape)

    def density_3d_auto(self, r_pc):
        """Safely choose a fast literature approximation or numerical Abel.

        Within ``1e-4 <= r/R_e <= 1e3`` this follows the density hybrid
        recommended by Vitral & Mamon (2021): official VM20bis for
        ``0.5 <= n <= 3.4`` and the Simonneau & Prada (2004) five-point
        approximation for ``3.4 < n <= 10``. Values outside those domains are
        evaluated by the numerical reference rather than extrapolated.
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
            result[numerical_mask] = self.density_3d_numerical(r_arr[numerical_mask])

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
        """Return the 3-D density using the requested deprojection method."""
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
