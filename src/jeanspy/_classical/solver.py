"""Jeans-equation solvers for the classical NumPy/SciPy backend."""

from __future__ import annotations

import multiprocessing as multi
import warnings

import numpy as np
from scipy import integrate
from scipy.constants import parsec

from ..dequad import dequad
from .core import Model
from .profiles import AnisotropyModel, DMModel, StellarModel


GMsun_m3s2 = 1.32712440018e20


def _projected_radii(R_pc):
    R = np.asarray(R_pc, dtype=float)
    if R.ndim > 1 or R.size == 0:
        raise ValueError("R_pc must be a scalar or nonempty one-dimensional array.")
    if not np.all(np.isfinite(R) & (R > 0)):
        raise ValueError("LOS solvers require finite R_pc > 0; the center R=0 is not supported.")
    return np.atleast_1d(R)


class DSphModel(Model):
    r"""Composite spherical Jeans model for dwarf spheroidal systems.

    Notes
    -----
    **Inputs and units.** submodels must contain StellarModel, DMModel and
    AnisotropyModel; ``vmem_kms`` sets mean velocity. ``sigmalos2_dequad`` uses
    n outer and ``n_kernel`` inner nodes; sigmalos2 dispatches to that same
    route. ``R_pc`` is scalar or nonempty 1-D projected radius; ``r_pc`` is
    intrinsic radius (pc).

    **Returns and shape.** sigmar2 and sigmat2 return intrinsic radial and
    one-component tangential variances in (km/s)^2. sigmalos2 and
    ``sigmalos2_dequad`` return LOS variance arrays with input shape, or a
    scalar for scalar input. ``sigmalos_dequad`` returns km/s.
    ``integrand_sigmalos2``(u,``R_pc``) has shape (``N_R``,``N_u``).

    **Validity.** Finite positive projected radii; no central-limit LOS solver.
    Tracer has vanishing outer pressure. Numerical orders and adaptive
    tolerances are part of the analysis configuration.

    **Errors.** Malformed/invalid projected radii or nonphysical mass, density
    and integrand values raise ValueError. Adaptive reference integration may
    issue SciPy integration warnings.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_spherical.py``
    """

    name = "DSphModel"
    required_param_names = ["vmem_kms"]
    required_models = {
        "StellarModel": StellarModel,
        "DMModel": DMModel,
        "AnisotropyModel": AnisotropyModel,
    }
    ncpu = multi.cpu_count()

    def _sigmar2(self, r_pc):
        density_3d = self["StellarModel"].density_3d
        enclosed_mass = self["DMModel"].enclosed_mass
        f = self["AnisotropyModel"].f

        def integrand(r):
            return (
                density_3d(r)
                * f(r)
                * GMsun_m3s2
                * enclosed_mass(r)
                / r**2
                / f(r_pc)
                / density_3d(r_pc)
                * 1e-6
                / parsec
            )

        value, _ = integrate.quad(integrand, r_pc, np.inf)
        return value

    def sigmar2(self, r_pc):
        r"""Return the radial velocity dispersion squared at ``r_pc``.

        Notes
        -----
        **Inputs and units.** ``r_pc`` is a scalar or NumPy array of positive
        intrinsic radii in pc.

        **Returns and shape.** Variance in (km/s)^2, following input shape; scalar
        input is a zero-dimensional NumPy array.

        **Validity.** Uses adaptive integration to infinite radius with vanishing
        outer pressure. The intrinsic helpers do not apply all LOS input validation
        checks.
        """
        return np.vectorize(self._sigmar2)(r_pc)

    def sigmat2(self, r_pc):
        r"""Return the tangential velocity dispersion squared at ``r_pc``.

        Notes
        -----
        **Inputs and units.** ``r_pc`` is a scalar or NumPy array of positive
        intrinsic radii in pc.

        **Returns and shape.** Variance in (km/s)^2, following input shape; scalar
        input is a zero-dimensional NumPy array.

        **Validity.** Uses adaptive integration to infinite radius with vanishing
        outer pressure. The intrinsic helpers do not apply all LOS input validation
        checks.
        """
        beta = self["AnisotropyModel"].beta(r_pc)
        return self.sigmar2(r_pc) * (1.0 - beta)

    def integrand_sigmalos2(self, u, R_pc, n_kernel=128):
        r"""Return the LOS-dispersion integrand.

        The integration variable is :math:`u=r/R`, with domain
        :math:`1 < u < \infty`.
        """
        R_pc = _projected_radii(R_pc)[:, np.newaxis]
        u = np.atleast_1d(np.asarray(u))[np.newaxis, :]

        density_3d = self["StellarModel"].density_3d
        density_2d = self["StellarModel"].density_2d
        enclosed_mass = self["DMModel"].enclosed_mass
        kernel = self["AnisotropyModel"].kernel
        r = R_pc * u
        mass = enclosed_mass(r)
        if not np.all(np.isfinite(mass) & (mass >= 0)):
            raise ValueError("Dark-matter enclosed mass must be finite and nonnegative")

        nu = density_3d(r)
        sigma = density_2d(R_pc)
        if not np.all(np.isfinite(nu) & (nu >= 0)) or not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError("Stellar density must be finite and nonnegative, with positive surface density.")

        value = (
            2.0
            * kernel(u, R_pc, n=n_kernel)
            / u
            * nu
            / sigma
            * GMsun_m3s2
            * mass
            / parsec
            * 1e-6
        )
        # DE nodes include u==1 and an extremely distant underflowed tracer
        # tail. Only these known zero-contribution locations may be discarded.
        value = np.where((u == 1) | (nu == 0), 0.0, value)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite LOS integrand outside a zero-density tail or endpoint.")
        return value

    def sigmalos2_dequad(
        self,
        R_pc,
        n=1024,
        n_kernel=128,
        ignore_RuntimeWarning=True,
    ):
        r"""Evaluate LOS variance for finite R_pc > 0 (scalar or 1-D array).

        R=0 requires a separate model-dependent central limit and is rejected.

        Notes
        -----
        **Inputs and units.** ``R_pc`` is positive finite scalar or nonempty
        one-dimensional pc array; n is the outer fixed-rule order and ``n_kernel``
        the anisotropy-kernel order.

        **Returns and shape.** Variance in (km/s)^2, or dispersion in km/s for
        ``sigmalos_dequad``; scalar for scalar input, otherwise (N,).
        """
        scalar_input = np.ndim(R_pc) == 0
        R_array = _projected_radii(R_pc)

        def func(u):
            return self.integrand_sigmalos2(u, R_array, n_kernel)

        with warnings.catch_warnings():
            if ignore_RuntimeWarning:
                warnings.simplefilter("ignore", RuntimeWarning)
            value = dequad(
                func,
                1,
                np.inf,
                axis=-1,
                n=n,
            )

        invalid = ~np.isfinite(value) | (np.asarray(value) < 0)
        if np.any(invalid):
            bad = R_array[invalid]
            raise ValueError(
                f"sigmalos2 is nonfinite or negative at R_pc = {bad} pc; "
                f"sigmalos2 = {np.asarray(value)[invalid]}; "
                f"current model parameters: {self.params_all}"
            )

        if scalar_input:
            return np.asarray(value).reshape(-1)[0]
        return value

    def sigmalos_dequad(
        self,
        R_pc,
        n=1024,
        n_kernel=128,
        ignore_RuntimeWarning=True,
    ):
        r"""Return the LOS velocity dispersion in km/s.

        Notes
        -----
        **Inputs and units.** ``R_pc`` is positive finite scalar or nonempty
        one-dimensional pc array; n is the outer fixed-rule order and ``n_kernel``
        the anisotropy-kernel order.

        **Returns and shape.** Variance in (km/s)^2, or dispersion in km/s for
        ``sigmalos_dequad``; scalar for scalar input, otherwise (N,).
        """
        return np.sqrt(
            self.sigmalos2_dequad(R_pc, n, n_kernel, ignore_RuntimeWarning)
        )

    def sigmalos2(
        self,
        R_pc,
        n=1024,
        n_kernel=128,
        ignore_RuntimeWarning=True,
    ):
        r"""Backend-neutral entry point for classical LOS dispersion squared.

        Notes
        -----
        **Inputs and units.** ``R_pc`` is positive finite scalar or nonempty
        one-dimensional pc array; n is the outer fixed-rule order and ``n_kernel``
        the anisotropy-kernel order.

        **Returns and shape.** Variance in (km/s)^2, or dispersion in km/s for
        ``sigmalos_dequad``; scalar for scalar input, otherwise (N,).
        """
        return self.sigmalos2_dequad(R_pc, n, n_kernel, ignore_RuntimeWarning)


__all__ = ["DSphModel", "GMsun_m3s2"]
