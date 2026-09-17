"""Physical profile components for the NumPy/SciPy backend."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1, k0

from ..dequad import dequad
from .._zhao import enclosed_mass as _zhao_mass, valid_domain as _zhao_valid
from .core import Model
from .jfactor import C_J, _ullio2016_inner_weight, _ullio2016_weight

_EXP_HALF_LIGHT_FACTOR = 1.67834699001666


def _jfactor_quad(integrand, lo, hi, **kwargs):
    """Do not return an unconverged or nonphysical quadrature as a J-factor."""
    result = quad(integrand, lo, hi, full_output=True, **kwargs)
    value, error = result[:2]
    if len(result) != 3 or not np.isfinite(value) or value < 0:
        detail = result[3] if len(result) > 3 else "nonfinite or negative integral"
        raise ValueError(f"J-factor quadrature failed: {detail}")
    return value, error


class StellarModel(Model):
    r"""Base class for projected/deprojected stellar-density models.

    Notes
    -----
    **Inputs and units.** Subclasses define ``density_2d``(``R_pc``),
    ``density_3d``(``r_pc``) and required parameters.
    density(``distance_from_center``, dimension) dispatches on '2d'/'3d'.
    ``density_2d_truncated``(``R_pc``, ``R_trunc_pc``) requires a scalar
    positive cutoff in pc.

    **Returns and shape.** Density in pc^-2/pc^-3; truncated surface density
    integrates to one within the cutoff and is zero outside. Shape follows
    radius input.

    **Validity.** A 3-D density is required by a Jeans solver. The truncated
    surface-density helper also requires ``cdf_R`` in the concrete profile.

    **Errors.** Abstract methods raise NotImplementedError; invalid dimension,
    radii or truncation raise ValueError.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_profiles.py``
    """

    name = "stellar Model"
    required_models = {}

    def density(self, distance_from_center, dimension):
        """Dispatch to the projected or intrinsic normalized tracer density.

        ``distance_from_center`` is a radius in pc. ``dimension`` must be
        '2d' or '3d', returning pc^-2 or pc^-3 respectively with the concrete
        profile's input shape. An unsupported dimension raises ValueError.
        """
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

        Notes
        -----
        **Inputs and units.** ``R_pc`` is a nonnegative finite scalar/array in pc;
        ``R_trunc_pc`` is a positive finite scalar cutoff in pc.

        **Returns and shape.** Normalized surface density in pc^-2 with radius
        shape, zero for R>``R_trunc_pc``.

        **Validity.** Requires a concrete ``density_2d`` and ``cdf_R`` with positive
        CDF at the cutoff.
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
        """Evaluate normalized surface density in pc^-2 at projected R_pc in pc.

        Subclasses define scalar/array support. This base method raises
        NotImplementedError.
        """
        raise NotImplementedError

    @abstractmethod
    def density_3d(self, r_pc):
        """Evaluate normalized intrinsic density in pc^-3 at r_pc in pc.

        Subclasses define scalar/array support. This base method raises
        NotImplementedError.
        """
        raise NotImplementedError


class PlummerModel(StellarModel):
    r"""Unit-integral Plummer tracer; re_pc is its projected half-light radius.

    Notes
    -----
    **Inputs and units.** ``re_pc`` (pc). Other constructor arguments follow
    Model.

    **Returns and shape.** ``density_2d`` and ``density_3d`` return pc^-2 and
    pc^-3 with input shape. ``cdf_R`` returns the dimensionless projected radial
    CDF; ``half_light_radius`` returns pc. ``mean_density_2d`` is mean surface
    density within R. ``logdensity_2d`` is the natural log of the surface
    density, not the radial PDF. ``density_2d_normalized_re``
    is the dimensionless ratio Sigma(R)/Sigma(re).

    **Validity.** Supply positive finite scales and real nonnegative radii. Use
    NumPy arrays for array inputs. The central density is finite; elementary
    formulas do not uniformly validate domains.

    **Errors.** Unknown parameter names raise ValueError in Model.update.
    Invalid values in elementary profile formulas can produce NaN/inf;
    successful construction alone does not validate a physical profile.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_profiles.py``
    """
    name = "Plummer Model"
    required_param_names = ["re_pc"]
    required_models = {}

    def density_2d(self, R_pc):
        r"""Evaluate normalized projected tracer density.

        Notes
        -----
        **Inputs and units.** ``R_pc`` is a scalar or NumPy array of projected radii
        in pc.

        **Returns and shape.** pc^-2 with input radius shape.
        """
        re_pc = self.params.re_pc
        return 1.0 / (1.0 + (R_pc / re_pc) ** 2) ** 2 / np.pi / re_pc**2

    def logdensity_2d(self, R_pc):
        r"""Evaluate natural log projected density.

        Notes
        -----
        **Inputs and units.** ``R_pc`` in pc; scalar or NumPy array.

        **Returns and shape.** Natural log of the numerical surface density in
        pc^-2, with input shape. This excludes the radial Jacobian 2\*pi\*R.
        """
        re_pc = self.params.re_pc
        return (
            -2.0 * np.log1p((R_pc / re_pc) ** 2)
            - np.log(np.pi)
            - 2.0 * np.log(re_pc)
        )

    def density_2d_normalized_re(self, R_pc):
        """Return the dimensionless ratio Sigma(R)/Sigma(re).

        ``R_pc`` is a scalar or NumPy array of projected radii in pc. Output
        shape follows the input; the stored re_pc must be positive.
        """
        re_pc = self.params.re_pc
        return 4.0 / (1.0 + (R_pc / re_pc) ** 2) ** 2

    def density_3d(self, r_pc):
        """Evaluate the unit-normalized intrinsic tracer density.

        Parameters
        ----------
        r_pc : float or numpy.ndarray
            Nonnegative intrinsic radius in pc; NumPy array inputs keep their shape.

        Returns
        -------
        float or numpy.ndarray
            Density in pc^-3 with the input radius shape.

        Notes
        -----
        The Plummer central density is finite. Stored tracer scales must be positive.
        This elementary NumPy formula does not uniformly validate input domains;
        no JAX physical-parameter differentiation is supported.
        """
        re_pc = self.params.re_pc
        return (3.0 / (4.0 * np.pi * re_pc**3)) / np.sqrt(
            1.0 + (r_pc / re_pc) ** 2
        ) ** 5

    def cdf_R(self, R_pc):
        r"""Return :math:`\int_0^R 2\pi R'\Sigma(R')\,dR'`.

        Notes
        -----
        **Inputs and units.** ``R_pc`` is a scalar or NumPy array of projected radii
        in pc.

        **Returns and shape.** Dimensionless cumulative probability with input
        radius shape.
        """
        re_pc = self.params.re_pc
        return 1.0 / (1.0 + (re_pc / R_pc) ** 2)

    def mean_density_2d(self, R_pc):
        r"""Average surface density inside a circular aperture.

        Notes
        -----
        **Inputs and units.** Nonnegative projected radius ``R_pc`` in pc, scalar or
        NumPy array.

        **Returns and shape.** ``1/(pi*(R_pc**2 + re_pc**2))``, in pc^-2 with
        radius shape. This expression has a finite value at R_pc=0.

        **Validity.** Nonnegative radii and a positive stored re_pc.
        """
        re_pc = self.params.re_pc
        return 1.0 / np.pi / (R_pc**2 + re_pc**2)

    def _half_light_radius(self, re_pc):
        return re_pc

    def half_light_radius(self):
        r"""Return the projected half-light radius.

        Notes
        -----
        **Inputs and units.** No arguments; reads the stored tracer scales.

        **Returns and shape.** Scalar radius in pc, equal to the stored ``re_pc``.
        """
        return self._half_light_radius(self.params.re_pc)


class ProjectedExponentialModel(StellarModel):
    r"""Stellar model with an exponential projected surface density.

    Notes
    -----
    **Inputs and units.** ``r_exp_pc`` is the scale in ``exp(-R/r_exp_pc)``
    in pc. The read-only ``re_pc`` property is the projected half-light radius,
    1.67834699001666 times this scale. Other constructor arguments follow Model.

    **Returns and shape.** ``density_2d`` and ``density_3d`` return pc^-2 and
    pc^-3 with input shape. ``cdf_R`` returns the dimensionless projected radial
    CDF; ``half_light_radius`` returns pc. ``mean_density_2d`` is mean surface
    density within R. ``logdensity_2d`` is the natural log of the surface
    density, not the radial PDF.

    **Validity.** Supply positive finite scales and real nonnegative radii. Use
    NumPy arrays for array inputs. A central cusp may diverge at zero. The older
    elementary density formulas do not uniformly validate domains.

    **Errors.** Unknown parameter names raise ValueError in Model.update.
    Invalid values in elementary profile formulas can produce NaN/inf;
    successful construction alone does not validate a physical profile.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_profiles.py``
    """

    name = "ProjectedExponentialModel"
    required_param_names = ["r_exp_pc"]
    required_models = {}

    @property
    def re_pc(self):
        """Projected half-light radius in pc, 1.67834699001666 * r_exp_pc."""
        return _EXP_HALF_LIGHT_FACTOR * self.params.r_exp_pc

    def density_2d(self, R_pc):
        r"""Evaluate normalized projected tracer density.

        Notes
        -----
        **Inputs and units.** ``R_pc`` is a scalar or NumPy array of projected radii
        in pc.

        **Returns and shape.** pc^-2 with input radius shape.
        """
        scale = self.params.r_exp_pc
        return np.exp(-R_pc / scale) / (2.0 * np.pi * scale**2)

    def logdensity_2d(self, R_pc):
        r"""Evaluate natural log projected density.

        Notes
        -----
        **Inputs and units.** ``R_pc`` in pc; scalar or NumPy array.

        **Returns and shape.** Natural log of the numerical surface density in
        pc^-2, with input shape. This excludes the radial Jacobian 2\*pi\*R.
        """
        scale = self.params.r_exp_pc
        return np.log(1.0 / (2.0 * np.pi)) - 2.0 * np.log(scale) - R_pc / scale

    def density_3d(self, r_pc):
        """Evaluate the unit-normalized intrinsic tracer density.

        Parameters
        ----------
        r_pc : float or numpy.ndarray
            Nonnegative intrinsic radius in pc; NumPy array inputs keep their shape.

        Returns
        -------
        float or numpy.ndarray
            Density in pc^-3 with the input radius shape.

        Notes
        -----
        The deprojected exponential density diverges at r_pc=0.
        Negative radii are outside the supported domain. Stored tracer scales must be positive.
        This elementary NumPy formula does not uniformly validate input domains;
        no JAX physical-parameter differentiation is supported.
        """
        scale = self.params.r_exp_pc
        return k0(r_pc / scale) / (2.0 * np.pi**2 * scale**3)

    def cdf_R(self, R_pc):
        r"""Evaluate projected probability inside a circular radius.

        Notes
        -----
        **Inputs and units.** ``R_pc`` is a scalar or NumPy array of projected radii
        in pc.

        **Returns and shape.** Dimensionless cumulative probability with input
        radius shape.
        """
        scale = self.params.r_exp_pc
        return 1.0 - np.exp(-R_pc / scale) * (1.0 + R_pc / scale)

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
        return self.cdf_R(R_pc) / (np.pi * R_pc**2)

    def half_light_radius(self):
        """Return the projected half-light radius in pc, equal to ``re_pc``."""
        return self.re_pc


class Uniform2dModel(StellarModel):
    r"""Unit-integral uniform projected disk with radius ``Rmax_pc`` in pc.

    ``density_2d(R_pc)`` returns 1/(pi*Rmax_pc**2) for 0 <= R <= Rmax_pc
    and zero outside; ``cdf_R`` returns the clipped dimensionless radial CDF.
    Outputs follow the scalar/array radius shape. The inherited
    ``density_2d_truncated`` can normalize an aperture inside the disk.

    The disk radius must be positive and finite; NaN radii raise ValueError.
    There is no three-dimensional tracer: ``density_3d`` raises
    NotImplementedError, so this profile cannot feed a 3-D Jeans solver.
    Half-light, mean-density and log-density methods are not provided.
    This is a stateful NumPy/SciPy model without JAX differentiation.
    See ``examples/docs_profiles.py``.
    """
    name = "uniform Model"
    required_param_names = ["Rmax_pc"]
    required_models = {}

    def density_2d(self, R_pc):
        r"""Evaluate normalized projected tracer density.

        Notes
        -----
        **Inputs and units.** ``R_pc`` is a scalar or NumPy array of projected radii
        in pc.

        **Returns and shape.** pc^-2 with input radius shape.
        """
        R, rmax = self._validated_radii(R_pc)
        return np.where((R >= 0) & (R <= rmax), 1.0 / (np.pi * rmax**2), 0.0)

    def density_3d(self, r_pc):
        """This projected uniform-disk model has no implemented 3-D density.

        The radius argument r_pc is in pc but is not evaluated. Every call
        raises NotImplementedError; there is no numerical return value.
        """
        raise NotImplementedError("Uniform2dModel has no 3-D density model.")

    def cdf_R(self, R_pc):
        r"""Evaluate projected probability inside a circular radius.

        Notes
        -----
        **Inputs and units.** ``R_pc`` is a scalar or NumPy array of projected radii
        in pc.

        **Returns and shape.** Dimensionless cumulative probability with input
        radius shape.
        """
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
    r"""Base class for NumPy/SciPy dark-matter density profiles.

    Notes
    -----
    **Inputs and units.** Subclasses supply ``mass_density_3d`` and
    ``enclosed_mass``/``enclosure_mass``. The base numerical J-factor methods
    require scalar ``dist_pc``, ``roi_deg`` (cone half-angle in degrees) and
    ``r_t_pc``. NFW overrides the spherical-aperture approximation and also supplies an
    Evans formula; these analytic methods support broadcastable geometry.

    **Returns and shape.** Mass in Msun; density in Msun/pc^3; J factor in GeV^2
    cm^-5.

    **Validity.** J-factor geometry requires a finite positive ``r_t_pc`` and an
    external observer for the full cone. The spherical-aperture method omits shells outside
    the spherical aperture; see the factors guide.

    **Errors.** Invalid scales/apertures, divergent cusps or failed adaptive
    quadrature raise ValueError.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_factors.py``
    """

    name = "DM Model"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.small_angle_limit_deg = 1.0

    @abstractmethod
    def mass_density_3d(self, r_pc):
        """Evaluate spherical halo density in Msun/pc^3 at radius r_pc in pc.

        This subclassing interface raises NotImplementedError. Concrete profiles
        specify array shapes, cutoff handling and behavior at a central cusp.
        """
        raise NotImplementedError

    def enclosed_mass(self, r_pc):
        r"""Return mass enclosed within ``r_pc``.

        ``enclosure_mass`` is retained on concrete models for compatibility
        with the original API.

        Notes
        -----
        **Inputs and units.** ``r_pc`` in pc, scalar or NumPy array; the Zhao
        implementation accepts ``n_steps``.

        **Returns and shape.** Msun within min(``r_pc``,``r_t_pc``), with input
        shape. ``enclosure_mass`` is the historical spelling.
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

        if small_angle and np.any(roi_deg > self.small_angle_limit_deg):
            raise ValueError(
                "Small-angle J-factor approximations require roi_deg <= "
                f"{self.small_angle_limit_deg} degrees."
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

    def validate_small_angle(self, roi_deg):
        """Validate cone half-angles for the small-angle J-factor methods.

        ``roi_deg`` is a scalar or broadcastable array in degrees. Returns None
        when all values are finite, positive and no larger than
        ``small_angle_limit_deg``. Otherwise raises ValueError.
        """
        try:
            roi_deg = np.asarray(roi_deg, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("roi_deg must be a finite positive number.") from exc
        if np.any(~np.isfinite(roi_deg)) or np.any(roi_deg <= 0):
            raise ValueError("roi_deg must be a finite positive number.")
        if np.any(roi_deg > self.small_angle_limit_deg):
            raise ValueError(
                "Small-angle J-factor approximations require roi_deg <= "
                f"{self.small_angle_limit_deg} degrees."
            )

    def jfactor_spherical_aperture(self, dist_pc, roi_deg=0.5):
        r"""Calculate the small-angle spherical-aperture approximation.

        Let ``R_max = dist_pc * sin(roi_deg)``.  This method integrates the
        spherical luminosity only out to ``min(R_max, r_t_pc)``.

        If ``R_max >= r_t_pc``, the aperture contains the entire truncated
        halo and this reduces to Ullio & Valli (2016), Eq. (B.10), with the
        halo boundary ``mathcal R = r_t_pc``.  If ``R_max < r_t_pc``, this is
        instead a spherical-aperture approximation: projected contributions
        from shells with ``R_max < r < r_t_pc`` are omitted.  Use
        :meth:`jfactor_cone` for the full finite-ROI geometry of
        Eqs. (B.8)--(B.9).

        Notes
        -----
        **Inputs and units.** ``dist_pc`` is observer distance in pc; ``roi_deg`` is
        cone half-angle in degrees. Distance, aperture and the stored ``r_t_pc``
        must all be scalars; array geometry raises ValueError.

        **Returns and shape.** A scalar J in GeV^2 cm^-5.

        **Validity.** Small-aperture spherical approximation; outer shells projected
        into the cone are omitted. Require a positive finite halo cutoff and a
        convergent inner cusp. Small-angle variants enforce the configured
        ``small_angle_limit_deg`` bound.
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

    def jfactor_cone(self, dist_pc, roi_deg=0.5):
        r"""Calculate the full finite-ROI Ullio & Valli (2016) J-factor.

        Unlike :meth:`jfactor_spherical_aperture`, this includes the projected
        contribution from shells with ``R_max < r < r_t_pc`` when the ROI is
        smaller than the truncated halo, following Eqs. (B.8)--(B.9).

        Notes
        -----
        **Inputs and units.** ``dist_pc`` is observer distance in pc; ``roi_deg`` is
        cone half-angle in degrees. Distance, aperture and the stored ``r_t_pc``
        must all be scalars; array geometry raises ValueError.

        **Returns and shape.** A scalar J in GeV^2 cm^-5.

        **Validity.** Full finite-distance cone; 0<``roi_deg``<=90,
        ``dist_pc``>``r_t_pc``. Require a positive finite halo cutoff and a
        convergent inner cusp. Small-angle variants enforce the configured
        ``small_angle_limit_deg`` bound.
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

        \rho(r)=\rho_s (r/r_s)^{-\gamma}
        [1+(r/r_s)^\alpha]^{-(\beta-\gamma)/\alpha}.

    Notes
    -----
    **Inputs and units.** ``rs_pc`` (pc), ``rhos_Msunpc3`` (Msun/pc^3), alpha/beta/gamma
    (dimensionless transition, outer and inner slopes), ``r_t_pc`` (pc).
    ``r_pc`` is scalar or an array in pc; ``n_steps`` controls numerical Zhao
    mass integration.

    **Returns and shape.** ``mass_density_3d`` returns Msun/pc^3 with input
    shape; ``enclosed_mass`` returns Msun inside min(``r_pc``, ``r_t_pc``).
    ``enclosure_mass`` is a historical alias. Density is zero outside ``r_t_pc``.

    **Validity.** Positive scales and cutoff; Zhao alpha>0 and gamma<3 for finite
    central mass; finite-radius mass does not require beta>3. Total untruncated
    mass can diverge. J-factor requires gamma<1.5.

    **Errors.** Invalid Zhao density/mass domains raise ValueError. A valid
    central cusp can return infinite density. J-factor methods validate geometry and
    quadrature separately.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_profiles.py``; ``examples/docs_factors.py``
    """

    name = "Zhao Model"
    required_param_names = ["rs_pc", "rhos_Msunpc3", "alpha", "beta", "gamma", "r_t_pc"]
    required_models = {}

    def _validate_jfactor_profile(self):
        super()._validate_jfactor_profile()
        if (not np.all(_zhao_valid(np.asarray(1.0), self.params, np))
                or np.any(np.asarray(self.params.gamma) >= 1.5)):
            raise ValueError("Finite Zhao J-factor requires positive scales, alpha > 0, and gamma < 1.5; "
                             "steeper central cusps have divergent annihilation luminosity")

    def mass_density_3d(self, r_pc):
        r"""Evaluate spherical halo density.

        Notes
        -----
        **Inputs and units.** ``r_pc`` in pc, scalar or NumPy array; reads the
        model's stored physical parameters.

        **Returns and shape.** Msun/pc^3 with input shape; zero for r > r_t_pc.
        The density includes the boundary r = r_t_pc; cusps can diverge at r=0.
        Invalid physical domains raise ValueError.
        """
        rs_pc = self.params.rs_pc
        rhos = self.params.rhos_Msunpc3
        alpha, beta, gamma = self.params.alpha, self.params.beta, self.params.gamma
        r = np.asarray(r_pc)
        if not np.all(_zhao_valid(r, self.params, np)):
            raise ValueError("Invalid Zhao density domain: require positive scales, "
                             "alpha > 0, gamma < 3 and nonnegative radii")
        inside = r <= self.params.r_t_pc
        x = np.where(inside, r / rs_pc, 1.0)
        with np.errstate(divide="ignore"):
            rho = rhos * x**(-gamma) * (1.0 + x**alpha)**(-(beta - gamma) / alpha)
        return np.where(inside, rho, 0.0)

    def enclosed_mass(self, r_pc, *, n_steps=128):
        r"""Finite-radius Zhao mass for alpha > 0 and gamma < 3, including beta <= 3.

        n_steps controls Gauss-Legendre nodes per regularized segment.

        Notes
        -----
        **Inputs and units.** ``r_pc`` in pc, scalar or NumPy array; the Zhao
        implementation accepts ``n_steps``.

        **Returns and shape.** Msun within min(``r_pc``,``r_t_pc``), with input
        shape. ``enclosure_mass`` is the historical spelling.
        """
        params = {k: getattr(self.params, k) for k in self.required_param_names}
        if not np.all(_zhao_valid(np.asarray(r_pc), params, np)):
            raise ValueError(
                "Invalid Zhao mass domain: require positive scales, alpha > 0, "
                "gamma < 3, finite slopes and nonnegative truncated radii"
            )
        return _zhao_mass(r_pc, params, xp=np, n_steps=n_steps)

    def enclosure_mass(self, r_pc, *, n_steps=128):
        r"""Evaluate halo mass inside a finite spherical radius.

        Notes
        -----
        **Inputs and units.** ``r_pc`` in pc, scalar or NumPy array; the Zhao
        implementation accepts ``n_steps``.

        **Returns and shape.** Msun within min(``r_pc``,``r_t_pc``), with input
        shape. ``enclosure_mass`` is the historical spelling.
        """
        return self.enclosed_mass(r_pc, n_steps=n_steps)


class NFWModel(DMModel):
    r"""Spherical NFW halo with a hard density cutoff at ``r_t_pc``.

    Parameters are ``rs_pc`` and ``r_t_pc`` in pc, and ``rhos_Msunpc3``
    in Msun/pc^3. Inside the cutoff, rho = rhos / (x*(1+x)**2), x=r/rs.
    ``mass_density_3d`` returns zero for r > r_t_pc and includes the cutoff
    boundary. The central density diverges. ``enclosed_mass`` (also spelled
    ``enclosure_mass``) gives the analytic mass inside min(r, r_t_pc), in Msun.
    Scalar/array outputs follow the input radius shape.

    Scales must be positive, rs and rhos finite; an infinite cutoff is allowed
    for density/mass at finite radii. Density validates this domain and raises
    ValueError on invalid values. The elementary mass formula does not
    uniformly validate inputs. A finite cutoff is required by J-factor methods.
    ``jfactor_cone`` integrates the truncated halo; ``jfactor_spherical_aperture``
    is an analytic aperture approximation. ``jfactor_small_angle_infinite_los``
    instead integrates the untruncated LOS profile with a capped projected
    aperture; it is not the J-factor of this hard-truncated density.

    This is a stateful NumPy/SciPy model without JAX differentiation.
    See ``examples/docs_profiles.py`` and ``examples/docs_factors.py``.
    """
    name = "NFW Model"
    required_param_names = ["rs_pc", "rhos_Msunpc3", "r_t_pc"]
    required_models = {}

    def _validate_jfactor_profile(self):
        super()._validate_jfactor_profile()
        if np.any(np.asarray(self.params.rs_pc) <= 0) or np.any(np.asarray(self.params.rhos_Msunpc3) <= 0):
            raise ValueError("NFW J-factor requires positive rs_pc and rhos_Msunpc3")

    def mass_density_3d(self, r_pc):
        r"""Evaluate spherical halo density.

        Notes
        -----
        **Inputs and units.** ``r_pc`` in pc, scalar or NumPy array; reads the
        model's stored physical parameters.

        **Returns and shape.** Msun/pc^3 with input shape; zero for r > r_t_pc.
        The density includes the boundary r = r_t_pc; cusps can diverge at r=0.
        Invalid physical domains raise ValueError.
        """
        rs_pc = self.params.rs_pc
        rhos = self.params.rhos_Msunpc3
        r = np.asarray(r_pc)
        domain = dict(self.params, alpha=1.0, beta=3.0, gamma=1.0)
        if not np.all(_zhao_valid(r, domain, np)):
            raise ValueError("Invalid NFW density domain: require positive scales "
                             "and nonnegative radii")
        inside = r <= self.params.r_t_pc
        x = np.where(inside, r / rs_pc, 1.0)
        with np.errstate(divide="ignore"):
            rho = rhos / x / (1.0 + x)**2
        return np.where(inside, rho, 0.0)

    def enclosure_mass(self, r_pc):
        r"""Evaluate halo mass inside a finite spherical radius.

        Notes
        -----
        **Inputs and units.** ``r_pc`` in pc, scalar or NumPy array; the Zhao
        implementation accepts ``n_steps``.

        **Returns and shape.** Msun within min(``r_pc``,``r_t_pc``), with input
        shape. ``enclosure_mass`` is the historical spelling.
        """
        threshold = 1e-7
        rs_pc = self.params.rs_pc
        rhos = self.params.rhos_Msunpc3
        r_t_pc = self.params.r_t_pc
        r_pc_trunc = np.minimum(np.asarray(r_pc), r_t_pc)
        x = r_pc_trunc / rs_pc
        value = np.log1p(x) - x / (1.0 + x)
        value = np.where(x < threshold, np.asarray(x) ** 2 / 2.0, value)
        return 4.0 * np.pi * rs_pc**3 * rhos * value

    def jfactor_spherical_aperture(self, dist_pc, roi_deg=0.5):
        r"""Evaluate the NFW spherical-aperture approximation analytically.

        The geometric interpretation is the same as
        :meth:`DMModel.jfactor_spherical_aperture`: for ``R_max >= r_t_pc`` the
        aperture encloses the full truncated halo and corresponds to the
        Eq. (B.10) limit; for ``R_max < r_t_pc`` it omits projected outer-shell
        contributions.  Use :meth:`jfactor_cone` for the full finite-ROI
        geometry.

        Notes
        -----
        **Inputs and units.** ``dist_pc`` is observer distance in pc; ``roi_deg`` is
        cone half-angle in degrees. This NFW analytic override supports mutually
        broadcastable distances, apertures and model parameters. The inherited
        full-cone ``jfactor_cone`` method requires scalar geometry.

        **Returns and shape.** J in GeV^2 cm^-5, with the broadcast input shape.

        **Validity.** Small-aperture spherical approximation; outer shells projected
        into the cone are omitted. Require a positive finite halo cutoff and a
        convergent inner cusp. Small-angle variants enforce the configured
        ``small_angle_limit_deg`` bound.
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

    def jfactor_small_angle_infinite_los(self, dist_pc, roi_deg=0.5):
        r"""Evaluate the small-angle, infinite-LOS Evans et al. (2016) formula.

        This historical approximation caps the *projected aperture* at r_t_pc;
        it does not truncate the density along the line of sight. For a halo
        truncated in three dimensions use jfactor_cone instead.

        Notes
        -----
        **Inputs and units.** ``dist_pc`` in pc and ``roi_deg`` in degrees, positive
        finite broadcastable values; stored NFW scales/cutoff.

        **Returns and shape.** J in GeV^2 cm^-5, with the broadcast input shape;
        scalar for scalar inputs.

        **Validity.** Small-angle formula: the projected aperture is capped at
        ``r_t_pc`` but the LOS density is untruncated. It is a different integral
        from the three-dimensionally truncated finite-cone factor.
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
    r"""Subclassing interface for spherical anisotropy.

    Notes
    -----
    **Inputs and units.** Implement beta(``r_pc``), f(``r_pc``) and
    kernel(u,``R_pc``,n) consistently; radii use pc and u=r/R.

    **Returns and shape.** Dimensionless beta/kernel and an
    arbitrary-normalization integrating factor f.

    **Validity.** Steady spherical Jeans closure; beta is distinct from
    axisymmetric ``beta_z``.

    **Errors.** Abstract methods raise NotImplementedError.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_profiles.py``
    """
    name = "AnisotropyModel"

    @abstractmethod
    def beta(self, r):
        """Evaluate dimensionless spherical anisotropy at radius r in pc.

        Subclasses must implement this interface; the base method raises
        NotImplementedError. Concrete models define scalar/array broadcasting.
        """
        raise NotImplementedError

    @abstractmethod
    def f(self, r):
        """Evaluate the Jeans integrating factor at radius r in pc.

        Its overall normalization cancels in the Jeans solution. Subclasses must
        implement this interface; the base method raises NotImplementedError.
        """
        raise NotImplementedError

    @abstractmethod
    def kernel(self, u, R, **kwargs):
        """Evaluate the dimensionless LOS kernel for u=r/R and projected radius R.

        Use u>=1, R>0 in pc and broadcastable arrays. Subclasses define numerical
        kwargs; the base method raises NotImplementedError.
        """
        raise NotImplementedError


class ConstantAnisotropyModel(AnisotropyModel):
    r"""Constant spherical anisotropy beta(r)=beta_ani.

    ``beta_ani`` is dimensionless and must be <1 for positive tangential dispersion.
    ``beta`` returns a scalar; ``f(r)`` is r**(2*beta_ani).
    Radii are in pc; f is an arbitrarily normalized Jeans integrating factor
    satisfying d ln(f)/d ln(r)=2*beta. The kernel is dimensionless and uses
    u=r/R >= 1 with positive R.

    ``kernel(u,R,**kwargs)`` uses an analytic hypergeometric expression,
    is independent of R and returns the u shape. Extra keywords are ignored.
    Elementary formulas do not uniformly validate physical domains; invalid
    inputs can produce nonfinite results. An anisotropy below 1 does not alone
    establish a positive phase-space distribution function.
    This stateful NumPy/SciPy API does not support JAX differentiation.
    See ``examples/docs_profiles.py``.
    """
    name = "ConstantAnisotropyModel"
    required_param_names = ["beta_ani"]
    required_models = {}

    def beta(self, r):
        r"""Evaluate spherical velocity anisotropy.

        Notes
        -----
        **Inputs and units.** r is a radius or NumPy radius array in pc.

        **Returns and shape.** Dimensionless beta; constant models may return a
        scalar.
        """
        del r
        return self.params.beta_ani

    def f(self, r):
        r"""Evaluate the radial Jeans integrating factor.

        Notes
        -----
        **Inputs and units.** r is a radius or NumPy radius array in pc.

        **Returns and shape.** An arbitrarily normalized integrating factor with
        radius shape.
        """
        return r ** (2.0 * self.params.beta_ani)

    def kernel(self, u, R, **kwargs):
        r"""Evaluate the spherical LOS projection kernel.

        Notes
        -----
        **Inputs and units.** u=r/R>=1 is dimensionless; R is projected radius in
        pc; broadcastable arrays. Extra kwargs are ignored by this analytic kernel.

        **Returns and shape.** Dimensionless projection kernel with the u shape; R is ignored.
        """
        del R, kwargs
        b = self.params.beta_ani
        u2 = u**2
        return np.sqrt(1.0 - 1.0 / u2) * (
            (1.5 - b) * u2 * hyp2f1(1.0, 1.5 - b, 1.5, 1.0 - u2) - 0.5
        )


class OsipkovMerrittModel(AnisotropyModel):
    r"""Osipkov--Merritt spherical anisotropy with an analytic kernel.

    ``r_a`` is a positive radius in pc. beta(r)=r**2/(r**2+r_a**2);
    f(r)=1+r**2/r_a**2. Both preserve the radius shape.
    Radii are in pc; f is an arbitrarily normalized Jeans integrating factor
    satisfying d ln(f)/d ln(r)=2*beta. The kernel is dimensionless and uses
    u=r/R >= 1 with positive R.

    ``kernel(u,R,**kwargs)`` returns the broadcast u/R shape from a closed
    form. Extra keywords are ignored; no quadrature order is needed.
    Elementary formulas do not uniformly validate physical domains; invalid
    inputs can produce nonfinite results. An anisotropy below 1 does not alone
    establish a positive phase-space distribution function.
    This stateful NumPy/SciPy API does not support JAX differentiation.
    See ``examples/docs_profiles.py``.
    """
    name = "OsipkovMerrittModel"
    required_param_names = ["r_a"]
    required_models = {}

    def beta(self, r):
        r"""Evaluate spherical velocity anisotropy.

        Notes
        -----
        **Inputs and units.** r is a radius or NumPy radius array in pc.

        **Returns and shape.** Dimensionless beta with radius shape.
        """
        r_a = self.params.r_a
        return r**2 / (r**2 + r_a**2)

    def f(self, r):
        r"""Evaluate the radial Jeans integrating factor.

        Notes
        -----
        **Inputs and units.** r is a radius or NumPy radius array in pc.

        **Returns and shape.** An arbitrarily normalized integrating factor with
        radius shape.
        """
        r_a = self.params.r_a
        return (r_a**2 + r**2) / r_a**2

    def kernel(self, u, R, **kwargs):
        r"""Evaluate the spherical LOS projection kernel.

        Notes
        -----
        **Inputs and units.** u=r/R>=1 is dimensionless; R is projected radius in
        pc; broadcastable arrays. Extra kwargs are ignored by this analytic kernel.

        **Returns and shape.** Dimensionless projection kernel with the broadcast
        shape.
        """
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
    r"""Baes--van Hese spherical anisotropy with numerical LOS quadrature.

    ``beta_0`` and ``beta_inf`` are dimensionless inner/outer anisotropies;
    ``r_a`` (pc) and ``eta`` are positive radius and sharpness. With
    t=(r/r_a)**eta, beta(r)=(beta_0+beta_inf*t)/(1+t). ``beta`` and ``f``
    preserve radius shape.
    Radii are in pc; f is an arbitrarily normalized Jeans integrating factor
    satisfying d ln(f)/d ln(r)=2*beta. The kernel is dimensionless and uses
    u=r/R >= 1 with positive R.

    ``kernel(u,R,n=128)`` flattens u and R and returns a grid with shape
    (R.size,u.size), including (1,1) for scalar inputs. n controls the
    double-exponential inner quadrature; increase it to check convergence.
    Elementary formulas do not uniformly validate physical domains; invalid
    inputs can produce nonfinite results. An anisotropy below 1 does not alone
    establish a positive phase-space distribution function.
    This stateful NumPy/SciPy API does not support JAX differentiation.
    See ``examples/docs_profiles.py``.
    """
    name = "BaesAnisotropyModel"
    required_param_names = ["beta_0", "beta_inf", "r_a", "eta"]
    required_models = {}

    def beta(self, r):
        r"""Evaluate spherical velocity anisotropy.

        Notes
        -----
        **Inputs and units.** r is a radius or NumPy radius array in pc.

        **Returns and shape.** Dimensionless beta with radius shape.
        """
        b0, binf = self.params.beta_0, self.params.beta_inf
        r_a, eta = self.params.r_a, self.params.eta
        x = np.power(r / r_a, eta)
        return (b0 + binf * x) / (1.0 + x)

    def f(self, r):
        r"""Evaluate the radial Jeans integrating factor.

        Notes
        -----
        **Inputs and units.** r is a radius or NumPy radius array in pc.

        **Returns and shape.** An arbitrarily normalized integrating factor with
        radius shape.
        """
        b0, binf = self.params.beta_0, self.params.beta_inf
        r_a, eta = self.params.r_a, self.params.eta
        x = np.power(r / r_a, eta)
        return np.power(r, 2.0 * b0) * np.power(
            1.0 + x, 2.0 * (binf - b0) / eta
        )

    def integrand_kernel(self, u_integ, R):
        """Evaluate the inner Baes LOS-kernel integrand.

        ``u_integ`` is the dimensionless integration radius r/R and must exceed
        one; R is the positive projected radius in pc. Broadcasting follows
        NumPy. The endpoint at one is singular and must be handled by quadrature.
        """
        u2_integ = u_integ**2
        r_integ = R * u_integ
        return (
            u_integ
            / np.sqrt(u2_integ - 1.0)
            * (1.0 - self.beta(r_integ) / u2_integ)
            / self.f(r_integ)
        )

    def kernel(self, u, R, **kwargs):
        r"""Evaluate the spherical LOS projection kernel.

        Notes
        -----
        **Inputs and units.** u=r/R>=1 is dimensionless; R is projected radius in
        pc; broadcastable arrays. The Baes numerical implementation takes fixed
        quadrature n through kwargs.

        **Returns and shape.** Dimensionless kernel on the (R.size, u.size) Cartesian grid,
        after flattening both inputs. Scalar inputs return shape (1,1).
        """
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
    "ProjectedExponentialModel",
    "NFWModel",
    "OsipkovMerrittModel",
    "PlummerModel",
    "StellarModel",
    "Uniform2dModel",
    "ZhaoModel",
]
