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


class DSphModel(Model):
    """Composite spherical Jeans model for dwarf spheroidal systems."""

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
        """Return the radial velocity dispersion squared at ``r_pc``."""
        return np.vectorize(self._sigmar2)(r_pc)

    def sigmat2(self, r_pc):
        """Return the tangential velocity dispersion squared at ``r_pc``."""
        beta = self["AnisotropyModel"].beta(r_pc)
        return self.sigmar2(r_pc) * (1.0 - beta)

    def integrand_sigmalos2(self, u, R_pc, n_kernel=128):
        r"""Return the LOS-dispersion integrand.

        The integration variable is :math:`u=r/R`, with domain
        :math:`1 < u < \infty`.
        """
        R_pc = np.atleast_1d(np.asarray(R_pc))[:, np.newaxis]
        u = np.atleast_1d(np.asarray(u))[np.newaxis, :]

        density_3d = self["StellarModel"].density_3d
        density_2d = self["StellarModel"].density_2d
        enclosed_mass = self["DMModel"].enclosed_mass
        kernel = self["AnisotropyModel"].kernel
        r = R_pc * u

        return (
            2.0
            * kernel(u, R_pc, n=n_kernel)
            / u
            * density_3d(r)
            / density_2d(R_pc)
            * GMsun_m3s2
            * enclosed_mass(r)
            / parsec
            * 1e-6
        )

    def sigmalos2_dequad(
        self,
        R_pc,
        n=1024,
        n_kernel=128,
        ignore_RuntimeWarning=True,
    ):
        """Evaluate the LOS velocity dispersion squared with DE quadrature."""
        scalar_input = np.ndim(R_pc) == 0
        R_array = np.atleast_1d(np.asarray(R_pc))

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
                replace_inf_to_zero=True,
                replace_nan_to_zero=True,
            )

        if np.any(value < 0):
            bad = R_array[np.asarray(value) < 0]
            raise ValueError(
                f"sigmalos2 is negative at R_pc = {bad} pc; "
                f"sigmalos2 = {np.asarray(value)[np.asarray(value) < 0]}; "
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
        """Return the LOS velocity dispersion in km/s."""
        return np.sqrt(
            self.sigmalos2_dequad(
                R_pc,
                n=n,
                n_kernel=n_kernel,
                ignore_RuntimeWarning=ignore_RuntimeWarning,
            )
        )

    def sigmalos2(
        self,
        R_pc,
        n=1024,
        n_kernel=128,
        ignore_RuntimeWarning=True,
    ):
        """Backend-neutral entry point for classical LOS dispersion squared."""
        return self.sigmalos2_dequad(
            R_pc,
            n=n,
            n_kernel=n_kernel,
            ignore_RuntimeWarning=ignore_RuntimeWarning,
        )


__all__ = ["DSphModel", "GMsun_m3s2"]
