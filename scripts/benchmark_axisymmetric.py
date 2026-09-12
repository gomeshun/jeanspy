#!/usr/bin/env python3
"""Record forward/autodiff latency; timing is not an accuracy or calibration test."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np

from jeanspy.axisymmetric import AxisymmetricDSphModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stars", type=int, default=8)
    parser.add_argument("--orders", type=int, nargs="+", default=[32, 64, 96])
    args = parser.parse_args()
    if args.stars < 1:
        parser.error("stars must be positive")
    from jeanspy.axisymmetric_numpyro import AxisymmetricDSphModel as JaxModel
    import jax
    import jax.numpy as jnp
    x = np.geomspace(10., 1200., args.stars)
    y = .4*x
    p = dict(re_pc=300., rs_pc=500., rhos_Msunpc3=.1, q=.7, Q=.8,
             alpha=2., beta=3., gamma=.8, beta_z=-.3, inclination=1.1, r_t_pc=3000.)
    def timed(function):
        start = time.perf_counter()
        value = jax.block_until_ready(function())
        return time.perf_counter()-start, value
    report = dict(platform=platform.platform(), backend=jax.default_backend(),
                  x64=bool(jax.config.jax_enable_x64), stars=args.stars, params=p,
                  versions={name: version(name) for name in ["numpy", "scipy", "jax", "jaxlib"]}, runs=[])
    for n in args.orders:
        classical, compiled = AxisymmetricDSphModel(n, n, n), JaxModel(n, n, n)
        reference_seconds, reference = timed(lambda: classical.sigmalos2(x, y, params=p))
        first_seconds, _ = timed(lambda: compiled.sigmalos2(x, y, params=p))
        value_seconds, value = timed(lambda: compiled.sigmalos2(x, y, params=p))
        grad = jax.jit(jax.grad(lambda params: jnp.sum(compiled.sigmalos2(x, y, params=params))))
        grad_first_seconds, _ = timed(lambda: grad(p))
        grad_seconds, derivatives = timed(lambda: grad(p))
        run = dict(order=n, numpy_seconds=reference_seconds, jax_first_seconds=first_seconds,
                    jax_warm_seconds=value_seconds, grad_first_seconds=grad_first_seconds,
                    grad_warm_seconds=grad_seconds,
                    backend_max_relative_difference=float(np.max(np.abs(np.asarray(value)/reference-1))),
                    gradient_finite=all(np.isfinite(v).all() for v in derivatives.values()))
        report["runs"].append(run)
        print(json.dumps(run), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
