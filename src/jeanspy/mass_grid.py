"""Quadrature for enclosed spherical masses on a shared grid of physical radii."""
from functools import lru_cache
import operator

import numpy as np


@lru_cache(maxsize=32)
def _unit_rule(order):
    order = operator.index(order)
    if order < 1:
        raise ValueError("order must be a positive integer")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1) / 2, weights / 2


class SphericalMassGrid:
    """Gauss panels in log radius for a fixed, nonempty grid of positive finite radii.

    Radii may repeat and have any shape. Supply the enclosed mass at the
    smallest radius separately so the central cusp can use its own quadrature.
    At `log_nodes`, evaluate dM/dln(r) = 4*pi*r**3*rho(r). Additional trailing
    axes can carry integrands such as derivatives at fixed physical radius.
    `integrate` returns the values in the input radius order and shape.
    The panel order controls accuracy across gaps in radius. Increase it and
    check convergence for sparse grids or rapidly changing density profiles.

    Pass numpy (default) or jax.numpy as xp. JAX radii must meet the same
    positive, finite precondition when traced. Geometry may be prepared once
    outside a parameter loop.
    """

    def __init__(self, radii, *, order=8, xp=np):
        radii = xp.asarray(radii)
        if radii.size == 0:
            raise ValueError("a nonempty radial grid is required")
        if xp is np and not np.all((radii > 0) & np.isfinite(radii)):
            raise ValueError("grid radii must be positive and finite")
        self._xp = xp
        self._shape = radii.shape
        flat = radii.reshape(-1)
        index = xp.argsort(flat)
        self._inverse = xp.argsort(index)
        sorted_r = flat[index]
        self.first_radius = sorted_r[0]
        logs = xp.log(sorted_r)
        widths = xp.diff(logs)
        self.max_log_step = xp.max(widths, initial=0.)
        nodes, weights = (xp.asarray(v, dtype=logs.dtype) for v in _unit_rule(order))
        self.log_nodes = logs[:-1, None] + widths[:, None] * nodes
        self._weights = widths[:, None] * weights

    def integrate(self, central, shell):
        """Accumulate the central value and shell integrals, including trailing moments."""
        xp = self._xp
        shell = xp.asarray(shell)
        if shell.shape[:2] != self.log_nodes.shape:
            raise ValueError("shell must start with the dimensions of log_nodes")
        trailing = shell.shape[2:]
        weights = self._weights.reshape(self._weights.shape + (1,) * len(trailing))
        steps = xp.sum(weights * shell, axis=1)
        start = xp.asarray(central).reshape((1,) + trailing)
        values = xp.cumsum(xp.concatenate((start, steps), axis=0), axis=0)
        return values[self._inverse].reshape(self._shape + trailing)
