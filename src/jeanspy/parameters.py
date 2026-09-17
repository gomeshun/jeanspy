"""Explicit sampling coordinates for NumPy/SciPy inference.

NumPyro's distribution-bearing specifications are in
``jeanspy.sampler_numpyro.ParameterSpec``. Neither interface infers a
transformation from the spelling of a sample name.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["SamplingParameter"]


@dataclass(frozen=True)
class SamplingParameter:
    """Map one sampled coordinate to a physical model parameter.

    Parameters
    ----------
    sample_name : str
        Coordinate name, matching one row in the finite-prior table.
    param_name : str
        Physical model parameter receiving the transformed value.
    transform : str, optional
        ``identity`` (default), ``pow10`` (10**x), ``one_minus_pow10``
        (1-10**x), or ``arccos`` (radians). Names never select a transform.

    Notes
    -----
    Prior bounds and photometric priors are evaluated in sampled coordinates.
    No Jacobian is added: a uniform bound on x with ``pow10`` defines a
    log-uniform physical parameter. Supply specifications in prior-table order.
    Invalid names or transform choices raise ValueError at construction.
    Conversion uses NumPy and is not a JAX tracing interface.

    Examples
    --------
    >>> SamplingParameter("log_radius", "re_pc", "pow10").to_physical(2.)
    np.float64(100.0)
    """

    sample_name: str
    param_name: str
    transform: str = "identity"

    def __post_init__(self):
        if any(not isinstance(name, str) or not name for name in
               (self.sample_name, self.param_name)):
            raise ValueError("Sampling and physical parameter names must be nonempty strings")
        if self.transform not in {"identity", "pow10", "one_minus_pow10", "arccos"}:
            raise ValueError("transform must be identity, pow10, one_minus_pow10 or arccos")

    def to_physical(self, value):
        """Convert a scalar/array, preserving shape; invalid values yield NaN/inf.

        Physical units are those of ``param_name``; arccos returns radians.
        The consuming likelihood is responsible for its physical-domain check.
        """
        value = np.asarray(value)
        with np.errstate(over="ignore", invalid="ignore"):
            if self.transform == "pow10":
                return np.power(10.0, value)
            if self.transform == "one_minus_pow10":
                return 1.0 - np.power(10.0, value)
            if self.transform == "arccos":
                return np.arccos(value)
        return value


def _validate_parameter_specs(specs, names):
    if specs is None:
        specs = tuple(SamplingParameter(name, name) for name in names)
    else:
        specs = tuple(specs)
    if not all(isinstance(spec, SamplingParameter) for spec in specs):
        raise TypeError("parameter_specs must contain SamplingParameter objects")
    if [spec.sample_name for spec in specs] != list(names):
        raise ValueError("parameter_specs sample names/order must match the prior table exactly")
    physical = [spec.param_name for spec in specs]
    if len(set(physical)) != len(physical):
        raise ValueError("Sampled physical parameter names must be unique")
    return specs


def _photometry_coordinate(specs, param_name="re_pc"):
    for index, spec in enumerate(specs):
        if spec.param_name == param_name and spec.transform == "pow10":
            return index
    raise ValueError(f"The photometry prior requires an explicit pow10 specification for {param_name} "
                     f"(a coordinate in log10({param_name}))")
