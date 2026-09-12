#!/usr/bin/env python3
"""Independent JAM-equation benchmark with inspectable MGE and quadrature errors.

Requires the optional benchmark extra (tested with jampy 8.1.4, mgefit 6.2.6).
The public JAM result is retained alongside a separately integrated JAM kernel:
the default JAM quadrature does not resolve these very broad MGEs reliably in
the tested environment. No third-party code or installed files are modified.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import re
import time
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"jeanspy-mpl"))

import numpy as np
from scipy.integrate import quad

from jeanspy.axisymmetric import AxisymmetricDSphModel, PlummerTracer, ZhaoHalo, G


COORDINATES = dict(x_pc=[10., 100., 300., 900., 0., 0., 0., 0., 70., -200.],
                   y_pc=[0., 0., 0., 0., 10., 100., 300., 900., -110., 400.])
CASES = [
    dict(re_pc=300., rs_pc=500., rhos_Msunpc3=.1, q=.65, Q=.55,
         alpha=1., beta=3., gamma=1., beta_z=-.3, inclination=1.1),
    dict(re_pc=300., rs_pc=500., rhos_Msunpc3=.1, q=.8, Q=1.3,
         alpha=2., beta=3., gamma=.5, beta_z=-.4, inclination=.8),
]
JAM_G = .004301  # documented in the 8.1.4 cylindrically aligned kernel


def density_mge(radius, density, count, inner, outer):
    import mgefit
    # Normalize both axes before the nonnegative fit for numerical conditioning.
    scale, amplitude = 300., float(np.max(density))
    result = mgefit.fit_1d(radius/scale, density/amplitude, ngauss=count,
                           inner_slope=inner, outer_slope=outer, quiet=True, plot=False)
    weights, widths = result.sol
    weights, widths = weights*scale*amplitude, widths*scale
    predicted = np.exp(-.5*(radius[:, None]/widths)**2) @ (weights/(np.sqrt(2*np.pi)*widths))
    relevant = (radius >= 1.) & (radius <= 1e4)
    return weights, widths, dict(
        max_relative_error=float(np.max(np.abs(predicted/density-1))),
        max_relative_error_1_to_10000_pc=float(np.max(np.abs(predicted[relevant]/density[relevant]-1))),
        fit_radius_pc=[float(radius[0]), float(radius[-1])],
        density_integrals=weights.tolist(), sigma_pc=widths.tolist())


def reference(params, ngauss):
    from jampy.axi.jam_axi_proj import jam_axi_proj, integrand_cyl_los, mge_surf
    tr = PlummerTracer(params["re_pc"], params["q"])
    halo = ZhaoHalo(params["rhos_Msunpc3"], params["rs_pc"], params["Q"],
                    params["alpha"], params["beta"], params["gamma"])
    rt = np.geomspace(.03, 3e5, 500)
    rh = np.geomspace(.05, 5e5, 600)
    c, s, tracer_fit = density_mge(rt, tr.density(rt, 0.), ngauss, 0, 5)
    h, sh, halo_fit = density_mge(rh, halo.density(rh, 0.), ngauss, halo.gamma, halo.beta)
    inc = params["inclination"]
    qp = lambda q: np.hypot(np.cos(inc), q*np.sin(inc))
    lum_density, halo_density = c/(np.sqrt(2*np.pi)*s), h/(np.sqrt(2*np.pi)*sh)
    lum_shape, halo_shape = np.full(s.size, tr.q), np.full(sh.size, halo.Q)
    lum_projected = c*tr.q/qp(tr.q)
    lum_qp, halo_qp = np.full(s.size, qp(tr.q)), np.full(sh.size, qp(halo.Q))
    anisotropy = np.full(s.size, params["beta_z"])
    x, y = (np.array(COORDINATES[key]) for key in ("x_pc", "y_pc"))

    raw = jam_axi_proj(lum_projected, s, lum_qp, h*halo.Q/qp(halo.Q), sh, halo_qp,
        np.rad2deg(inc), 0., .648/np.pi, x, y, align="cyl", beta=anisotropy,
        analytic_los=True, interp=False, ml=1., sigmapsf=0., pixsize=0., plot=False, quiet=True)
    values, refinement = [], []
    for xx, yy in zip(x, y):
        kernel_args = (lum_density, s, lum_shape, halo_density, sh, halo_shape,
                       xx, yy, inc, anisotropy, "zz")
        # Resolve narrow features at small u caused by large Gaussian scale
        # ratios. This is JAM's independent analytic LOS kernel, not JeansPy's
        # homoeoidal/vertical/LOS integration or its radial derivative.
        def log_integrand(logu):
            u = np.exp(logu)
            return float(integrand_cyl_los(np.array([u]), *kernel_args)[0])*u
        surface = mge_surf(xx, yy, lum_projected, s, lum_qp)
        coarse = quad(log_integrand, -35., 0., epsabs=0., epsrel=1e-8, limit=300)[0]
        fine = quad(log_integrand, -40., 0., epsabs=0., epsrel=1e-10, limit=400)[0]
        values.append(fine/surface*G/JAM_G)
        refinement.append(abs(coarse/fine-1))
    return dict(sigma2=values, public_jam_sigma2=(raw.model**2*G/JAM_G).tolist(),
                quadrature_refinement_max=float(max(refinement)), tracer_mge=tracer_fit, halo_mge=halo_fit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from jampy.axi.jam_axi_proj import integrand_cyl_los
    start = time.perf_counter()
    source = Path(inspect.getsourcefile(integrand_cyl_los))
    recorded_g = re.search(r"\bG\s*=\s*([\d.eE+-]+)", inspect.getsource(integrand_cyl_los))
    if recorded_g is None or float(recorded_g.group(1)) != JAM_G:
        raise RuntimeError("JAM's gravitational constant changed; review the benchmark normalization")
    package = Path(inspect.getsourcefile(AxisymmetricDSphModel)).parent
    result = dict(
        reference="Cappellari (2008), cylindrical JAM equation (28), evaluated by jampy",
        references=["https://arxiv.org/abs/0806.0042", "https://pypi.org/project/jampy/8.1.4/"],
        versions={p: version(p) for p in ["numpy", "scipy", "jampy", "mgefit"]},
        kernel_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        jeanspy_source_sha256={name: hashlib.sha256((package/name).read_bytes()).hexdigest()
                               for name in ["axisymmetric.py", "_axisymmetric_params.py"]},
        gravitational_constants=dict(jeanspy=G, jampy=JAM_G),
        coordinate_units="pc", moment_units="(km/s)^2", **COORDINATES,
        acceptance=dict(jeanspy_vs_resolved_jam_rtol=.003,
                        mge_refinement_rtol=.003, jam_quadrature_refinement_rtol=1e-7), cases=[])
    passed = True
    for p in CASES:
        coarse, fine = reference(p, 32), reference(p, 48)
        expected = np.asarray(fine["sigma2"])
        x, y = COORDINATES["x_pc"], COORDINATES["y_pc"]
        actual = AxisymmetricDSphModel(128, 128, 128).sigmalos2(x, y, params=p)
        numerical = AxisymmetricDSphModel(96, 96, 96).sigmalos2(x, y, params=p)
        error = float(np.max(np.abs(actual/expected-1)))
        mge_error = float(np.max(np.abs(np.asarray(coarse["sigma2"])/expected-1)))
        ok = error < .003 and mge_error < .003 and fine["quadrature_refinement_max"] < 1e-7
        case = dict(params=p, reference=fine, sigma2=actual.tolist(),
                     max_relative_error=error, mge_refinement_max=mge_error,
                     jeanspy_order_refinement_max=float(np.max(np.abs(numerical/actual-1))),
                     public_jam_difference_max=float(np.max(np.abs(np.asarray(fine["public_jam_sigma2"])/expected-1))),
                     passed=ok)
        result["cases"].append(case)
        passed &= ok
        print(f"Q={p['Q']}: resolved JAM difference {error:.3g}; MGE refinement {mge_error:.3g}; pass={ok}", flush=True)
    result.update(passed=passed, elapsed_seconds=time.perf_counter()-start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    if not passed:
        raise SystemExit("Axisymmetric JAM benchmark did not meet its declared tolerances; see the JSON report")


if __name__ == "__main__":
    main()
