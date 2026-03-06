from __future__ import annotations

import dataclasses
import jax.numpy as jnp
import jax
from jax.scipy.special import beta, betainc, erf, gamma, gammainc
from typing import Tuple
from abc import ABC, abstractmethod
from functools import partial
from scipy.integrate import quad

from .hyp2f1_jax import hyp2f1_1b_3half



# Constants from original model.py
GMsun_m3s2 = 1.32712440018e20
R_trunc_pc = 1866.

# Physical constants approximation (JAX-compatible)
# These would need proper JAX-compatible implementation
# For now, using approximated values from the original
solar_mass_kg = 1.9884e30
parsec = 3.0857e16  # meters
degree = jnp.pi/180
physical_constants_electron_volt_kg = 1.782661907e-36
physical_constants_electron_volt_m = 1.239841984e-6

kg_eV = 1./physical_constants_electron_volt_kg
im_eV = 1./physical_constants_electron_volt_m
C0 = (solar_mass_kg*kg_eV)**2*((1./parsec)*im_eV)**5
C1 = (1e9)**2 * (1e2*im_eV)**5
C_J = C0/C1


# -----------------------------------------------------------------------------
# Experimental: simple OO-style models (no dataclass / no PyTree)
# -----------------------------------------------------------------------------
# Motivation:
# - Keep a convenient OO API (initialize once, call many methods that share params).
# - Avoid forcing a dataclass/PyTree design before the composite-model story is settled.
#
# Important rule for NumPyro + gradient-based inference:
# - If a parameter is sampled by NumPyro (numpyro.sample), instantiate the model *inside*
#   the NumPyro model function using that sampled value.
# - Do not mutate model attributes after construction.


class StellarModelOO(ABC):
    """OO-style stellar model base class.

    This version intentionally does NOT implement PyTree registration.
    """

    def density(self, distance_from_center: jnp.ndarray, dimension: str):
        if dimension == "2d":
            return self.density_2d(distance_from_center)
        if dimension == "3d":
            return self.density_3d(distance_from_center)
        raise ValueError(f"Unknown dimension: {dimension!r}")

    def density_2d_truncated(self, R_pc: jnp.ndarray, R_trunc_pc: float):
        r"""Truncated 2D density.

        Note:
            \int_0^{R_trunc} 2\pi R density_2d_truncated(R,R_trunc) = 1
        """
        return self.density_2d(R_pc) / self.cdf_R(jnp.asarray(R_trunc_pc))

    @abstractmethod
    def density_2d(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def cdf_R(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        """Cumulative distribution function for projected radius R."""
        raise NotImplementedError


class PlummerModelOO(StellarModelOO):
    """Plummer profile (OO-style).

    Parameters
    ----------
    re_pc:
        Projected half-light radius in pc.
    """

    def __init__(self, re_pc):
        self.re_pc = re_pc

    def density_2d(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc = self.re_pc
        return 1 / (1 + (R_pc / re_pc) ** 2) ** 2 / jnp.pi / re_pc**2

    def logdensity_2d(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc = self.re_pc
        return -jnp.log1p((R_pc / re_pc) ** 2) * 2 - jnp.log(jnp.pi) - jnp.log(re_pc) * 2

    def density_2d_normalized_re(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc = self.re_pc
        return 4 / (1 + (R_pc / re_pc) ** 2) ** 2

    def density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc = self.re_pc
        return (3 / 4 / jnp.pi / re_pc**3) / jnp.sqrt(1 + (r_pc / re_pc) ** 2) ** 5

    def cdf_R(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc = self.re_pc
        return 1 / (1 + (re_pc / R_pc) ** 2)


def _constant_anisotropy_kernel_series(
    u: jnp.ndarray,
    beta_ani: jnp.ndarray,
    *,
    n_terms: int = 96,
    w_switch: float = 0.7,
) -> jnp.ndarray:
    r"""Kernel K(u) for the constant-anisotropy model (series backend).

    Uses the stable transformation (valid for z = 1-u^2 <= 0):

        2F1(1, 3/2-β; 3/2; 1-u^2) = u^{-2} * 2F1(1, β; 3/2; 1-1/u^2)

    so we only need 2F1(1, β; 3/2; w) with w in [0, 1).

    This backend forces the fixed-term power series implementation.
    """
    u2 = u**2
    w = 1.0 - 1.0 / u2
    hyp_w = hyp2f1_1b_3half(beta_ani, w, method="series", n_terms=n_terms, w_switch=w_switch)
    return jnp.sqrt(1 - 1 / u2) * ((1.5 - beta_ani) * hyp_w - 0.5)


def constant_anisotropy_kernel(
    u: jax.Array | jnp.ndarray | float,
    beta_ani: jax.Array | jnp.ndarray | float,
    *,
    method: str = "series",
    n_terms: int = 96,
) -> jnp.ndarray:
    """Kernel K(u) for the constant-anisotropy model.

    Parameters
    ----------
    u:
        Dimensionless radius ratio (u = r/R). Domain is typically u >= 1.
    beta_ani:
        Constant anisotropy parameter β.
    method:
        - "series": fixed-term power series approximation (reverse-mode differentiable).
        - "jax_hyp2f1": routes to the dedicated robust implementation (auto-switched).
        - "scipy_hyp2f1": uses scipy.special.hyp2f1 (host callback; comparison only).
    n_terms:
        Number of terms for the "series" method.
    """
    beta_ani = jnp.asarray(beta_ani)
    u = jnp.asarray(u)
    u2 = u**2

    if method == "series":
        return _constant_anisotropy_kernel_series(u, beta_ani, n_terms=n_terms)

    if method == "jax_hyp2f1":
        # Backward-compat: previously this relied on jax.scipy.special.hyp2f1 via a custom VJP.
        # We now route it to the dedicated robust implementation.
        w = 1.0 - 1.0 / u2
        hyp_w = hyp2f1_1b_3half(beta_ani, w, method="auto", n_terms=n_terms)
        return jnp.sqrt(1 - 1 / u2) * ((1.5 - beta_ani) * hyp_w - 0.5)

    if method == "scipy_hyp2f1":
        # WARNING: host-side; cannot be JIT/grad/vmap'd. Use only for offline comparisons.
        import numpy as np
        import scipy.special as _scipy_special

        # Ensure float64 evaluation for a reliable baseline (the JAX arrays are often float32).
        u_np = np.asarray(u, dtype=np.float64)
        beta_np = np.asarray(beta_ani, dtype=np.float64)
        # Use the same stable w-transform as the JAX implementation.
        w_np = 1.0 - 1.0 / (u_np**2)
        hyp_w_np = _scipy_special.hyp2f1(1.0, beta_np, 1.5, w_np)
        hyp_w = jnp.asarray(hyp_w_np)
        return jnp.sqrt(1 - 1 / u2) * ((1.5 - beta_ani) * hyp_w - 0.5)

    raise ValueError(f"Unknown method: {method!r}")


def compare_constant_anisotropy_kernel(
    *,
    beta_ani: float,
    u_min: float = 1.0 + 1e-3,
    u_max: float = 50.0,
    n_u: int = 256,
    baseline: str = "scipy_hyp2f1",
    n_terms_list: tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128, 160, 192, 256),
) -> dict:
    """Offline helper to choose a reasonable n_terms for the series approximation.

    Returns a dict with:
      - u
      - baseline_kernel
      - errors: {n_terms: {"max_abs": ..., "max_rel": ...}}

    Notes
    -----
    - baseline="scipy_hyp2f1" is recommended.
    - This function is intended for interactive exploration (not JIT).
    """
    u = jnp.linspace(u_min, u_max, n_u)
    base = constant_anisotropy_kernel(u, beta_ani, method=baseline)
    errors: dict[int, dict[str, float]] = {}

    base_abs = jnp.abs(base)
    rel_denom = jnp.maximum(base_abs, 1e-30)
    for n_terms in n_terms_list:
        approx = constant_anisotropy_kernel(u, beta_ani, method="series", n_terms=int(n_terms))
        diff = approx - base
        max_abs = float(jnp.max(jnp.abs(diff)))
        max_rel = float(jnp.max(jnp.abs(diff) / rel_denom))
        errors[int(n_terms)] = {"max_abs": max_abs, "max_rel": max_rel}

    return {
        "u": u,
        "baseline": baseline,
        "baseline_kernel": base,
        "errors": errors,
    }


def calibrate_constant_anisotropy_kernel_n_terms(
    *,
    beta_values: tuple[float, ...] = (-10.0, -5.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9, 0.99),
    u_min: float = 1.0 + 1e-3,
    u_max: float = 1e3,
    n_u: int = 256,
    baseline: str = "scipy_hyp2f1",
    n_terms_list: tuple[int, ...] = (8, 12, 16, 20, 24, 32, 48, 64),
) -> dict:
    """Offline helper to choose n_terms across a wide beta/u domain.

    It computes, for each n_terms, the worst-case max_abs/max_rel over the supplied
    beta_values and u grid.

    This is meant to be run interactively (not under JIT).
    """
    u = jnp.linspace(u_min, u_max, n_u)
    summary: dict[int, dict[str, float]] = {}

    for n_terms in n_terms_list:
        worst_abs = 0.0
        worst_rel = 0.0
        worst_at = float("nan")
        for beta_ani in beta_values:
            base = constant_anisotropy_kernel(u, beta_ani, method=baseline)
            approx = constant_anisotropy_kernel(u, beta_ani, method="series", n_terms=int(n_terms))
            diff = approx - base
            base_abs = jnp.abs(base)
            rel_denom = jnp.maximum(base_abs, 1e-30)
            max_abs = float(jnp.max(jnp.abs(diff)))
            max_rel = float(jnp.max(jnp.abs(diff) / rel_denom))
            if max_abs > worst_abs or max_rel > worst_rel:
                worst_abs = max(worst_abs, max_abs)
                worst_rel = max(worst_rel, max_rel)
                worst_at = float(beta_ani)
        summary[int(n_terms)] = {"worst_max_abs": worst_abs, "worst_max_rel": worst_rel, "worst_beta": worst_at}

    return {"u": u, "baseline": baseline, "beta_values": beta_values, "summary": summary}


class DMModelOO(ABC):
    """OO-style dark matter model base class (no PyTree registration)."""

    @abstractmethod
    def mass_density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def enclosure_mass(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        """Enclosed mass M(<r) in Msun."""
        raise NotImplementedError


class NFWModelOO(DMModelOO):
    """Navarro-Frenk-White profile (OO-style).

    Parameters
    ----------
    rs_pc:
        Scale radius in pc.
    rhos_Msunpc3:
        Density normalization in Msun/pc^3.
    r_t_pc:
        Truncation radius in pc (used for enclosure_mass truncation).
    """

    def __init__(self, rs_pc, rhos_Msunpc3, r_t_pc):
        self.rs_pc = rs_pc
        self.rhos_Msunpc3 = rhos_Msunpc3
        self.r_t_pc = r_t_pc

    def mass_density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        x = r_pc / self.rs_pc
        return self.rhos_Msunpc3 / x / (1 + x) ** 2

    def enclosure_mass(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        threshold = 1e-7  # avoid underflow for very small x
        r_pc_trunc = jnp.minimum(r_pc, self.r_t_pc)
        x = r_pc_trunc / self.rs_pc
        is_small = x < threshold

        # (1/(1+x) - 1 + log(1+x)) is stable for moderate x; Taylor for small x.
        ret = jnp.where(is_small, x**2 / 2, 1 / (1 + x) - 1 + jnp.log1p(x))
        return (4.0 * jnp.pi * self.rs_pc**3 * self.rhos_Msunpc3) * ret


class ZhaoModelOO(DMModelOO):
    """Zhao (alpha-beta-gamma) profile (OO-style)."""

    def __init__(self, rs_pc, rhos_Msunpc3, a, b, g, r_t_pc):
        self.rs_pc = rs_pc
        self.rhos_Msunpc3 = rhos_Msunpc3
        self.a = a
        self.b = b
        self.g = g
        self.r_t_pc = r_t_pc

    def mass_density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        x = r_pc / self.rs_pc
        return self.rhos_Msunpc3 * jnp.power(x, -self.g) * jnp.power(1 + jnp.power(x, self.a), -(self.b - self.g) / self.a)

    def enclosure_mass(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        r_pc_trunc = jnp.minimum(r_pc, self.r_t_pc)
        x = jnp.power(r_pc_trunc / self.rs_pc, self.a)
        argbeta0 = (3 - self.g) / self.a
        argbeta1 = (self.b - 3) / self.a
        return (
            (4.0 * jnp.pi * self.rs_pc**3 * self.rhos_Msunpc3 / self.a)
            * beta(argbeta0, argbeta1)
            * betainc(argbeta0, argbeta1, x / (1 + x))
        )


class AnisotropyModelOO(ABC):
    """OO-style anisotropy model base class (no PyTree registration)."""

    @abstractmethod
    def beta(self, r: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def f(self, r: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def kernel(self, u: jnp.ndarray, R: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError


class ConstantAnisotropyModelOO(AnisotropyModelOO):
    """Constant anisotropy model (OO-style)."""

    def __init__(self, beta_ani):
        self.beta_ani = beta_ani

    def beta(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.beta_ani * jnp.ones_like(r)

    def f(self, r: jnp.ndarray) -> jnp.ndarray:
        return r ** (2 * self.beta_ani)

    def kernel(self, u: jnp.ndarray, R: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Kernel function K(u) for LOSVD.

        NOTE: This is kept numerically equivalent to the existing JAX dataclass version,
        but will be converted to JAX-only special functions in a later pass.
        """
        method = "series" if ("method" not in kwargs) else str(kwargs["method"])
        n_terms = 96 if ("n_terms" not in kwargs) else int(kwargs["n_terms"])
        return constant_anisotropy_kernel(u, self.beta_ani, method=method, n_terms=n_terms)


class OsipkovMerrittModelOO(AnisotropyModelOO):
    """Osipkov-Merritt anisotropy model (OO-style)."""

    def __init__(self, r_a):
        self.r_a = r_a

    def beta(self, r: jnp.ndarray) -> jnp.ndarray:
        return r**2 / (r**2 + self.r_a**2)

    def f(self, r: jnp.ndarray) -> jnp.ndarray:
        return (self.r_a**2 + r**2) / self.r_a**2

    def kernel(self, u: jnp.ndarray, R: jnp.ndarray, **kwargs) -> jnp.ndarray:
        u_a = self.r_a / R
        u2_a = u_a**2
        u2 = u**2
        sqrt_term = jnp.sqrt((u2 - 1) / (u2_a + 1))
        arctan_term = jnp.arctan(sqrt_term)
        sqrt_term2 = jnp.sqrt(1 - 1 / u2)
        return (u2 + u2_a) * (u2_a + 0.5) / (u * (u2_a + 1) ** 1.5) * arctan_term - sqrt_term2 / 2 / (u2_a + 1)


class DSphModelOO:
    """Composite dwarf spheroidal model (OO-style).

    This class exists primarily to make "model composition" ergonomic while staying
    compatible with JAX autodiff.

    Notes
    -----
    - Keep this object immutable in practice (no attribute mutation after init).
    - When using NumPyro, instantiate this inside the NumPyro model function.
    """

    def __init__(
        self,
        stellar: StellarModelOO,
        dm: DMModelOO,
        anisotropy: AnisotropyModelOO,
        vmem_kms,
    ):
        self.stellar = stellar
        self.dm = dm
        self.anisotropy = anisotropy
        self.vmem_kms = vmem_kms

    def dm_mass_density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        return self.dm.mass_density_3d(r_pc)

    def dm_enclosure_mass(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        return self.dm.enclosure_mass(r_pc)

    def stellar_density_2d(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        return self.stellar.density_2d(R_pc)

    def stellar_density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        return self.stellar.density_3d(r_pc)

    def v_circ2(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        """Circular velocity squared v_c^2(r).

        Units
        -----
        This is a placeholder convenience method. The current file already defines
        physical constants, but a consistent unit system is still being settled.

        For now we return G * M(<r) / r in "(GMsun/pc)" units.
        """
        return self.dm.enclosure_mass(r_pc) / r_pc



def physmodel(cls: type) -> type:
    """Convert a plain class to (1) `dataclass`, (2) JAX PyTree in one shot."""
    cls = dataclasses.dataclass(cls)

    # --- PyTree handlers ------------------------------------------------------
    def _flatten(self):
        return (dataclasses.astuple(self), None)

    @classmethod
    def _unflatten(c_, aux, children):
        return c_(*children)
    
    cls.tree_flatten = _flatten
    cls.tree_unflatten = _unflatten

    # register after attaching methods so they exist during registration
    jax.tree_util.register_pytree_node_class(cls)
    return cls



@physmodel
class StellarModel:
    def density(self,distance_from_center,dimension):
        if dimension == "2d":
            return self.density_2d(distance_from_center)
        elif dimension == "3d":
            return self.density_3d(distance_from_center)
    def density_2d_truncated(self,R_pc,R_trunc_pc):
        """
        Truncated 2D density. Note that
            \\int_0^{R_trunc} 2\\pi R density_2d_truncated(R,R_trunc) = 1 .
        """
        return self.density_2d(R_pc)/self.cdf_R(R_trunc_pc)
    
    @abstractmethod
    def density_2d(self,R_pc) -> jnp.ndarray:
        pass

    @abstractmethod
    def density_3d(self,r_pc) -> jnp.ndarray:
        pass

    @abstractmethod
    def cdf_R(self,R_pc) -> jnp.ndarray:
        """Cumulative distribution function for radius R."""
        pass



@physmodel
class PlummerModel(StellarModel):
    re_pc: float  # projected half-light radius in pc

    def density_2d(self,R_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc= self.re_pc
        return 1/(1+(R_pc/re_pc)**2)**2 /jnp.pi/re_pc**2


    def logdensity_2d(self,R_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc= self.re_pc
        return -jnp.log1p((R_pc/re_pc)**2)*2 -jnp.log(jnp.pi) -jnp.log(re_pc)*2

    def density_2d_normalized_re(self,R_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc= self.re_pc
        return 4/(1+(R_pc/re_pc)**2)**2

    def density_3d(self,r_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc= self.re_pc
        return (3/4/jnp.pi/re_pc**3)/jnp.sqrt(1+(r_pc/re_pc)**2)**5

    def cdf_R(self,R_pc: jnp.ndarray) -> jnp.ndarray:
        """
        cdf_R(R) = \\int_0^R \\dd{R'} 2\\pi R' \\Sigma(R')
        """
        re_pc= self.re_pc
        return 1/(1+(re_pc/R_pc)**2)

    def mean_density_2d(self,R_pc: jnp.ndarray) -> jnp.ndarray:
        """
        return the mean density_2d in R < R_pc with the weight 2*pi*R
        mean_density_2d = \\frac{\\int_\\RoIR \\dd{R} 2\\pi R \\Sigma(R)}{\\int_\\RoIR \\dd{R} 2\\pi R}
            = \\frac{cdf_R(R)}{\\pi R^2}
        """
        re_pc= self.re_pc
        return 1/jnp.pi/(R_pc**2+re_pc**2)

    def _half_light_radius(self,re_pc: float) -> float:
        """
        Half-light-raduis means that the radius in which the half of all stars are include
        """
        return re_pc

    def half_light_radius(self) -> float:
        """
        Half-light-raduis means that the radius in which the half of all stars are include
        """
        return self._half_light_radius(self.re_pc)

# k0 function approximation - Modified Bessel function of the second kind of order 0
# For JAX compatibility, we'll use an approximation
def k0_approx(x):
    """Approximation of Modified Bessel function K_0(x) for x > 0"""
    # For small x, K_0(x) ~ -ln(x/2) - gamma
    # For large x, K_0(x) ~ sqrt(pi/(2x)) * exp(-x)
    gamma_euler = 0.5772156649015329
    small_x = x < 2.0
    
    result = jnp.where(small_x, 
                     -jnp.log(x/2.0) - gamma_euler,
                     jnp.sqrt(jnp.pi/(2*x)) * jnp.exp(-x))
    return result



@physmodel
class SersicModel(StellarModel):
    re_pc: float  # projected half-light radius in pc
    n: float      # Sersic index
    
    @property 
    def b_CB(self):
        """Approximation by Eq.(18) of Ciotti and Bertin (1999), [arXiv:astro-ph/9911078]
        It is valid for n > 0.5."""
        if self.n <= 0.5:
            raise ValueError("Sersic index n must be greater than 0.5 for the approximation to be valid.")
        n = self.n
        return 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2) + 131/(1148175*n**3) - 2194697/(30690717750*n**4)
    
    @property
    def b(self):
        """For simplicity, use CB approximation instead of interpolation"""
        return self.b_CB
    
    @property
    def norm(self):
        n = self.n
        return jnp.pi*self.re_pc**2 * jnp.power(self.b, -2*n) * gamma(2*n+1)
    
    def density_2d(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-self.b * jnp.power(R_pc/self.re_pc, 1/self.n)) / self.norm
    
    def density_2d_normalized_re(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-self.b * (jnp.power(R_pc/self.re_pc, 1/self.n) - 1))
    
    def cdf_R(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        """
        cdf_R(R) = \\int_0^R \\dd{R'} 2\\pi R' \\Sigma(R')
        """
        n = self.n
        return gammainc(2*n, self.b * jnp.power(R_pc/self.re_pc, 1/n))
    
    def mean_density_2d(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        """
        return the mean density_2d in R < R_pc with the weight 2*pi*R
        """
        return self.cdf_R(R_pc) / jnp.pi / R_pc**2
    
    @property
    def p_LGM(self):
        n = self.n
        return 1 - 0.6097/n + 0.05463/n**2
    
    @property
    def norm_3d(self):
        Rhalf = self.re_pc
        n = self.n
        b = self.b_CB
        p = self.p_LGM
        ind = (3-p)*n
        return 4 * jnp.pi * Rhalf**3 * n * gamma(ind) / b**ind
    
    def density_3d_LGM(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        p = self.p_LGM
        n = self.n
        b = self.b_CB
        x = (r_pc/self.re_pc)
        return x**(-p) * jnp.exp(-b * x**(1/n)) / self.norm_3d
    
    def density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        # Using LGM approximation
        return self.density_3d_LGM(r_pc)
    
    def half_light_radius(self) -> float:
        return self.re_pc


@physmodel
class Exp2dModel(StellarModel):
    """Stellar model whose 2D (projected, surface) density is given by the exponential model."""
    re_pc: float  # projected half-light radius in pc
    
    @property
    def R_exp_pc(self):
        return self.re_pc / 1.67834699001666
    
    def density_2d(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc = self.R_exp_pc
        return (1./2/jnp.pi/re_pc**2) * jnp.exp(-R_pc/re_pc)
    
    def logdensity_2d(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc = self.R_exp_pc
        return jnp.log(1./2/jnp.pi) - jnp.log(re_pc)*2 + (-R_pc/re_pc)
    
    def density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc = self.R_exp_pc
        return (1./2/jnp.pi**2/re_pc**3) * k0_approx(r_pc/re_pc)
    
    def cdf_R(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        """
        cdf_R(R) = \\int_0^R \\dd{R'} 2\\pi R' \\Sigma(R')
        """
        re_pc = self.R_exp_pc
        return 1. - jnp.exp(-R_pc/re_pc) * (1 + R_pc/re_pc)
    
    def mean_density_2d(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        return self.cdf_R(R_pc) / jnp.pi / R_pc**2
    
    def _half_light_radius(self, re_pc: float) -> float:
        return 1.67834699001666 * self.R_exp_pc
    
    def half_light_radius(self) -> float:
        return self._half_light_radius(self.re_pc)


@physmodel
class Exp3dModel(StellarModel):
    """Stellar model whose 3D (deprojected) density is given by the exponential model."""
    re_pc: float  # projected half-light radius in pc
    
    def density_2d(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc = self.re_pc
        return (1./2/jnp.pi/re_pc**2) * jnp.exp(-R_pc/re_pc)
    
    def density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc = self.re_pc
        return (1./2/jnp.pi**2/re_pc**3) * k0_approx(r_pc/re_pc)
    
    def cdf_R(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        """
        cdf_R(R) = \\int_0^R \\dd{R'} 2\\pi R' \\Sigma(R')
        """
        re_pc = self.re_pc
        return 1. - jnp.exp(-R_pc/re_pc) * (1 + R_pc/re_pc)
    
    def mean_density_2d(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        re_pc = self.re_pc
        return self.cdf_R(R_pc) / jnp.pi / R_pc**2
    
    def half_light_radius(self) -> float:
        return 1.67834699001666 * self.re_pc


@physmodel
class Uniform2dModel(StellarModel):
    """Uniform 2D stellar model."""
    Rmax_pc: float  # maximum radius in pc
    
    def density_2d(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        return 1./(jnp.pi * self.Rmax_pc**2) * jnp.ones_like(R_pc)
    
    def density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        # Not implemented for uniform 2D model
        raise NotImplementedError("density_3d not implemented for Uniform2dModel")
    
    def cdf_R(self, R_pc: jnp.ndarray) -> jnp.ndarray:
        return (R_pc/self.Rmax_pc)**2


# DMModel base class
@physmodel
class DMModel:
    """Base class for Dark Matter models."""
    
    def assert_roi_is_enough_small(self, roi_deg):
        roi_deg_max_warning = 1.0
        assert jnp.all(roi_deg <= roi_deg_max_warning)
    
    @abstractmethod
    def mass_density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        pass
    
    def jfactor_ullio2016_simple(self, dist_pc: float, roi_deg: float = 0.5):
        """Calculate J-factor of DM profile using Eq.(B.10) in [arXiv:1603.07721]."""
        self.assert_roi_is_enough_small(roi_deg)
        roi_pc = dist_pc * jnp.sin(jnp.deg2rad(roi_deg))
        
        # This would require numerical integration in JAX
        # For now, returning a placeholder
        # func = lambda r: r**2 * self.mass_density_3d(r)**2
        # integ = integrate(func, 0, roi_pc)  # Would need JAX-compatible integration
        # j = 4 * jnp.pi / dist_pc**2 * integ * C_J
        raise NotImplementedError("Numerical integration needed for J-factor calculation")


@physmodel 
class ZhaoModel(DMModel):
    """Zhao dark matter profile model."""
    rs_pc: float          # scale radius in pc
    rhos_Msunpc3: float   # density normalization in Msun/pc^3
    a: float              # inner slope parameter
    b: float              # outer slope parameter  
    g: float              # transition parameter
    r_t_pc: float         # truncation radius in pc
    
    def mass_density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        x = r_pc / self.rs_pc
        a = self.a
        b = self.b
        g = self.g
        return self.rhos_Msunpc3 * jnp.power(x, -g) * jnp.power(1 + jnp.power(x, a), -(b-g)/a)

    def enclosure_mass(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        a = self.a
        b = self.b
        g = self.g
        r_t_pc = self.r_t_pc
        # truncation
        r_pc_trunc = jnp.where(r_pc > r_t_pc, r_t_pc, r_pc)
        # auxiliary variable
        x = jnp.power(r_pc_trunc/self.rs_pc, a)
        argbeta0 = (3-g)/a
        argbeta1 = (b-3)/a
        # return the mass enclosed within r_pc
        return (4.*jnp.pi*self.rs_pc**3 * self.rhos_Msunpc3/a) * beta(argbeta0, argbeta1) * betainc(argbeta0, argbeta1, x/(1+x))


@physmodel
class NFWModel(DMModel):
    """Navarro-Frenk-White dark matter profile model."""
    rs_pc: float          # scale radius in pc
    rhos_Msunpc3: float   # density normalization in Msun/pc^3
    r_t_pc: float         # truncation radius in pc
    
    def mass_density_3d(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        x = r_pc / self.rs_pc
        return self.rhos_Msunpc3 / x / (1 + x)**2
    
    def enclosure_mass(self, r_pc: jnp.ndarray) -> jnp.ndarray:
        threshold = 1e-7  # threshold to avoid underflow
        r_t_pc = self.r_t_pc
        # truncation
        r_pc_trunc = jnp.where(r_pc > r_t_pc, r_t_pc, r_pc)
        x = r_pc_trunc / self.rs_pc
        
        # Taylor expansion for small x to avoid numerical issues
        is_small = x < threshold
        
        # Main calculation: (1/(1+x) - 1 + log(1+x))
        ret = jnp.where(is_small,
                      x**2/2,  # Series expansion up to second order
                      1/(1+x) - 1 + jnp.log(1+x))
        
        return (4.*jnp.pi*self.rs_pc**3 * self.rhos_Msunpc3) * ret
    
    def jfactor_ullio2016_simple(self, dist_pc: float, roi_deg: float = 0.5):
        self.assert_roi_is_enough_small(roi_deg)
        roi_pc = dist_pc * jnp.sin(jnp.deg2rad(roi_deg))
        r_max_pc = jnp.minimum(roi_pc, self.r_t_pc)
        c_max = r_max_pc / self.rs_pc
        
        j = C_J * 4 * jnp.pi * self.rs_pc**3 * self.rhos_Msunpc3**2 / dist_pc**2  # normalization
        j *= (1-1/(1+c_max)**3)/3 + ((self.rs_pc/dist_pc)**2 * c_max**3/(1+c_max)**3)/9  # approximation
        return j


# AnisotropyModel base class
@physmodel
class AnisotropyModel:
    """Base class for anisotropy models."""
    
    @abstractmethod
    def beta(self, r: jnp.ndarray) -> jnp.ndarray:
        """Anisotropy parameter as a function of radius."""
        pass
    
    @abstractmethod
    def f(self, r: jnp.ndarray) -> jnp.ndarray:
        """Auxiliary function f(r) for anisotropy calculations."""
        pass
    
    @abstractmethod
    def kernel(self, u: jnp.ndarray, R: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Kernel function K(u) for LOSVD calculations."""
        pass


@physmodel
class ConstantAnisotropyModel(AnisotropyModel):
    """Constant anisotropy model."""
    beta_ani: float | None = None  # constant anisotropy parameter
    beta_const: float | None = None  # backward-compatible alias

    def __post_init__(self) -> None:
        if self.beta_ani is None and self.beta_const is None:
            raise ValueError("Either beta_ani or beta_const must be provided.")
        if self.beta_ani is None:
            self.beta_ani = self.beta_const
        if self.beta_const is None:
            self.beta_const = self.beta_ani
    
    def beta(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.beta_ani * jnp.ones_like(r)
    
    def f(self, r: jnp.ndarray) -> jnp.ndarray:
        return r ** (2 * self.beta_ani)
    
    def kernel(self, u: jnp.ndarray, R: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        kernel function K(u). LOSVD is given by
        
            sigmalos2(R) = 2 * \\int_1^\\infty du \\nu_\\ast(uR)/\\Sigma_\\ast(R) * GM(uR) * K(u)/u.
        """
        method = "series" if ("method" not in kwargs) else str(kwargs["method"])
        n_terms = 96 if ("n_terms" not in kwargs) else int(kwargs["n_terms"])
        return constant_anisotropy_kernel(u, self.beta_ani, method=method, n_terms=n_terms)


@physmodel
class OsipkovMerrittModel(AnisotropyModel):
    """Osipkov-Merritt anisotropy model."""
    r_a: float  # anisotropy radius in pc
    
    def beta(self, r: jnp.ndarray) -> jnp.ndarray:
        return r**2 / (r**2 + self.r_a**2)
    
    def f(self, r: jnp.ndarray) -> jnp.ndarray:
        return (self.r_a**2 + r**2) / self.r_a**2
    
    def kernel(self, u: jnp.ndarray, R: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        u, R: 1d array
        """
        u_a = self.r_a / R
        u2_a = u_a**2
        u2 = u**2
        
        # Vectorized computation
        sqrt_term = jnp.sqrt((u2-1)/(u2_a+1))
        arctan_term = jnp.arctan(sqrt_term)
        sqrt_term2 = jnp.sqrt(1-1/u2)
        
        kernel = ((u2+u2_a)*(u2_a+0.5)/(u*(u2_a+1)**1.5) * arctan_term - 
                 sqrt_term2/2/(u2_a+1))
        return kernel


@physmodel
class BaesAnisotropyModel(AnisotropyModel):
    """Baes anisotropy model."""
    beta_0: float     # central anisotropy
    beta_inf: float   # asymptotic anisotropy
    r_a: float        # anisotropy scale radius in pc
    eta: float        # anisotropy transition parameter
    
    def beta(self, r: jnp.ndarray) -> jnp.ndarray:
        x = jnp.power(r/self.r_a, self.eta)
        return (self.beta_0 + self.beta_inf * x) / (1 + x)
    
    def f(self, r: jnp.ndarray) -> jnp.ndarray:
        x = jnp.power(r/self.r_a, self.eta)
        return jnp.power(r, 2*self.beta_0) * jnp.power(1+x, 2*(self.beta_inf-self.beta_0)/self.eta)
    
    def integrand_kernel(self, u_integ: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
        """
        u = r/R,
        us = r_a/R
        """
        u2_integ = u_integ**2
        r_integ = R * u_integ
        return (u_integ / jnp.sqrt(u2_integ - 1) * 
                (1 - self.beta(r_integ) / u2_integ) / self.f(r_integ))
    
    def kernel(self, u: jnp.ndarray, R: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        kernel function K(u). LOSVD is given by
        
            sigmalos2(R) = 2 * \\int_1^\\infty du \\nu_\\ast(uR)/\\Sigma_\\ast(R) * GM(uR) * K(u)/u.
            
        # u: ndarray, shape = (n_u,)
        # R: ndarray, shape = (n_R,)
        
        return: ndarray, shape = (n_R,n_u)
        """
        n = 128 if ("n" not in kwargs) else kwargs["n"]
        
        # For JAX compatibility, we'll use scipy.integrate.quad for now
        # This breaks JAX purity but provides correct results
        def compute_integral_for_single_u_R(u_val, R_val):
            def integrand(u_integ):
                return float(self.integrand_kernel(jnp.array(u_integ), jnp.array(R_val)))
            
            result, _ = quad(integrand, 1.0, float(u_val), limit=n)
            return result
        
        # Vectorize the computation
        if jnp.ndim(u) == 0:
            u = jnp.array([u])
        if jnp.ndim(R) == 0:
            R = jnp.array([R])
            
        results = []
        for R_val in R:
            row_results = []
            for u_val in u:
                integral_val = compute_integral_for_single_u_R(u_val, R_val)
                f_val = self.f(R_val * u_val)
                kernel_val = integral_val * f_val / u_val
                row_results.append(kernel_val)
            results.append(row_results)
        
        return jnp.array(results)
