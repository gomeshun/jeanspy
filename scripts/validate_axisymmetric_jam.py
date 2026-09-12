#!/usr/bin/env python3
"""Independent JAM-equation benchmark with inspectable MGE and quadrature errors.

Requires the optional benchmark extra (tested with jampy 8.1.4, mgefit 6.2.6).
Compare the public analytic-LOS path with independent quadrature of the same
JAM kernel. Record the numerical-LOS path and its resolution dependence too:
in jampy 8.1.4, interp=False overrides analytic_los=True. Verify the effective
path from the returned velocity tensor, not just the requested flags.
No third-party code or installed files are modified.
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


def reference(params, ngauss, *, coefficients=None, diagnostics=False):
    from jampy.axi.jam_axi_proj import jam_axi_proj, integrand_cyl_los, mge_surf
    from jampy.util.quad1d import quad1d
    tr = PlummerTracer(params["re_pc"], params["q"])
    halo = ZhaoHalo(params["rhos_Msunpc3"], params["rs_pc"], params["Q"],
                    params["alpha"], params["beta"], params["gamma"])
    rt = np.geomspace(.03, 3e5, 500)
    rh = np.geomspace(.05, 5e5, 600)
    if coefficients is None:
        _, _, tracer_fit = density_mge(rt, tr.density(rt, 0.), ngauss, 0, 5)
        _, _, halo_fit = density_mge(rh, halo.density(rh, 0.), ngauss, halo.gamma, halo.beta)
    else:
        tracer_fit, halo_fit = coefficients["tracer_mge"], coefficients["halo_mge"]
    c, s = (np.asarray(tracer_fit[key]) for key in ("density_integrals", "sigma_pc"))
    h, sh = (np.asarray(halo_fit[key]) for key in ("density_integrals", "sigma_pc"))
    inc = params["inclination"]
    qp = lambda q: np.hypot(np.cos(inc), q*np.sin(inc))
    lum_density, halo_density = c/(np.sqrt(2*np.pi)*s), h/(np.sqrt(2*np.pi)*sh)
    lum_shape, halo_shape = np.full(s.size, tr.q), np.full(sh.size, halo.Q)
    lum_projected = c*tr.q/qp(tr.q)
    lum_qp, halo_qp = np.full(s.size, qp(tr.q)), np.full(sh.size, qp(halo.Q))
    anisotropy = np.full(s.size, params["beta_z"])
    x, y = (np.array(COORDINATES[key]) for key in ("x_pc", "y_pc"))

    def public_call(interp, nrad=20, nang=10, nlos=1500, epsrel=1e-2):
        # With these ten positions, the analytic path evaluates each position
        # directly: psf_conv bypasses its output grid when nrad*nang > x.size.
        # interp=False at the public entry point would instead switch off
        # analytic_los before psf_conv is called.
        if interp and nrad*nang <= x.size:
            raise ValueError("Require direct-position evaluation for the analytic JAM comparison")
        settings = dict(analytic_los=True, interp=interp, nrad=nrad, nang=nang,
                        nlos=nlos, epsrel=epsrel)
        raw = jam_axi_proj(lum_projected, s, lum_qp, h*halo.Q/qp(halo.Q), sh, halo_qp,
            np.rad2deg(inc), 0., .648/np.pi, x, y, align="cyl", beta=anisotropy,
            ml=1., sigmapsf=0., pixsize=0., plot=False, quiet=True, **settings)
        effective_path = "analytic_los" if raw.vel2 is None else "numerical_los"
        expected_path = "analytic_los" if interp else "numerical_los"
        if effective_path != expected_path:
            raise RuntimeError(f"JAM path changed: expected {expected_path}, got {effective_path}; "
                               "review this benchmark for the installed jampy version")
        return dict(requested_settings=settings, effective_path=effective_path,
                    sigma2=(raw.model**2*G/JAM_G).tolist())

    public_analytic = public_call(True)
    values, refinement, direct_values, split_values, direct_error_bounds = [], [], [], [], []
    for xx, yy in zip(x, y):
        kernel_args = (lum_density, s, lum_shape, halo_density, sh, halo_shape,
                       xx, yy, inc, anisotropy, "zz")
        # Independent change of variables for JAM's analytic LOS kernel.
        # This verifies its integral without reusing JeansPy's nested solver;
        # it is not evidence that JAM's own quad1d cannot resolve this kernel.
        def log_integrand(logu):
            u = np.exp(logu)
            return float(integrand_cyl_los(np.array([u]), *kernel_args)[0])*u
        surface = mge_surf(xx, yy, lum_projected, s, lum_qp)
        coarse = quad(log_integrand, -35., 0., epsabs=0., epsrel=1e-8, limit=300)[0]
        fine = quad(log_integrand, -40., 0., epsabs=0., epsrel=1e-10, limit=400)[0]
        values.append(fine/surface*G/JAM_G)
        refinement.append(abs(coarse/fine-1))
        if diagnostics:
            direct = quad1d(integrand_cyl_los, [0., 1.], singular=True,
                            epsrel=1e-5, args=kernel_args)
            if direct.status != 0:
                raise RuntimeError("JAM quad1d did not report successful integration")
            split = quad(lambda u: float(integrand_cyl_los(np.array([u]), *kernel_args)[0]),
                         0., 1., points=np.geomspace(1e-8, .1, 15),
                         epsabs=0., epsrel=1e-10, limit=500)[0]
            direct_values.append(float(direct.integ)/surface*G/JAM_G)
            direct_error_bounds.append(float(abs(direct.errbnd/direct.integ)))
            split_values.append(split/surface*G/JAM_G)

    relative_difference = lambda v: float(np.max(np.abs(np.asarray(v)/values-1)))
    public_analytic["max_relative_difference"] = relative_difference(public_analytic["sigma2"])
    result = dict(sigma2=values, public_analytic_los=public_analytic,
                  quadrature_refinement_max=float(max(refinement)),
                  mge_components=dict(requested=ngauss, tracer=len(s), halo=len(sh)),
                  tracer_mge=tracer_fit, halo_mge=halo_fit)
    if diagnostics:
        numerical = [public_call(False), public_call(False, nrad=40, nang=20),
                     public_call(False, nrad=80, nang=40, nlos=3000, epsrel=1e-6)]
        for run in numerical:
            run["max_relative_difference"] = relative_difference(run["sigma2"])
        result.update(numerical_los_refinement=numerical,
            direct_quad1d=dict(sigma2=direct_values, epsrel=1e-5, status=0,
                              max_relative_difference=relative_difference(direct_values),
                              estimated_relative_error_max=max(direct_error_bounds)),
            split_u_quad=dict(sigma2=split_values, epsrel=1e-10,
                              max_relative_difference=relative_difference(split_values)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-mge", type=Path,
                        help="Reuse recorded MGE coefficients for the same cases and coordinates")
    args = parser.parse_args()
    from jampy.axi.jam_axi_proj import integrand_cyl_los
    start = time.perf_counter()
    source = Path(inspect.getsourcefile(integrand_cyl_los))
    recorded_g = re.search(r"\bG\s*=\s*([\d.eE+-]+)", inspect.getsource(integrand_cyl_los))
    if recorded_g is None or float(recorded_g.group(1)) != JAM_G:
        raise RuntimeError("JAM's gravitational constant changed; review the benchmark normalization")
    package = Path(inspect.getsourcefile(AxisymmetricDSphModel)).parent
    reused = None
    if args.reuse_mge is not None:
        reused = json.loads(args.reuse_mge.read_text())
        if (any(reused[key] != value for key, value in COORDINATES.items())
                or [case["params"] for case in reused["cases"]] != CASES):
            raise ValueError("Recorded MGE cases and coordinates must match this benchmark")
    result = dict(
        schema_version=2,
        reference="Cappellari (2008), cylindrical JAM equation (28), evaluated by jampy",
        references=["https://arxiv.org/abs/0806.0042", "https://pypi.org/project/jampy/8.1.4/"],
        versions={p: version(p) for p in ["numpy", "scipy", "jampy", "mgefit"]},
        kernel_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_mge_report_sha256=(hashlib.sha256(args.reuse_mge.read_bytes()).hexdigest()
                                  if args.reuse_mge is not None else None),
        jeanspy_source_sha256={name: hashlib.sha256((package/name).read_bytes()).hexdigest()
                               for name in ["axisymmetric.py", "_axisymmetric_params.py"]},
        gravitational_constants=dict(jeanspy=G, jampy=JAM_G),
        coordinate_units="pc", moment_units="(km/s)^2", **COORDINATES,
        acceptance=dict(jeanspy_vs_resolved_jam_rtol=.003,
                        mge_refinement_rtol=.003, jam_quadrature_refinement_rtol=1e-7,
                        public_analytic_los_rtol=1e-7, independent_kernel_quadrature_rtol=1e-7,
                        refined_numerical_los_rtol=.003), cases=[])
    passed = True
    for index, p in enumerate(CASES):
        saved = reused["cases"][index] if reused is not None else {}
        coarse = reference(p, 32, coefficients=saved.get("coarse_reference"))
        fine = reference(p, 48, coefficients=saved.get("reference"), diagnostics=True)
        expected = np.asarray(fine["sigma2"])
        x, y = COORDINATES["x_pc"], COORDINATES["y_pc"]
        actual = AxisymmetricDSphModel(128, 128, 128).sigmalos2(x, y, params=p)
        numerical = AxisymmetricDSphModel(96, 96, 96).sigmalos2(x, y, params=p)
        error = float(np.max(np.abs(actual/expected-1)))
        mge_error = float(np.max(np.abs(np.asarray(coarse["sigma2"])/expected-1)))
        numerical_errors = [run["max_relative_difference"] for run in fine["numerical_los_refinement"]]
        ok = (error < .003 and mge_error < .003 and fine["quadrature_refinement_max"] < 1e-7
              and fine["public_analytic_los"]["max_relative_difference"] < 1e-7
              and fine["direct_quad1d"]["max_relative_difference"] < 1e-7
              and fine["split_u_quad"]["max_relative_difference"] < 1e-7
              and numerical_errors[0] > numerical_errors[1] > numerical_errors[2]
              and numerical_errors[2] < .003)
        case = dict(params=p, reference=fine, coarse_reference=coarse, sigma2=actual.tolist(),
                     max_relative_error=error, mge_refinement_max=mge_error,
                     jeanspy_order_refinement_max=float(np.max(np.abs(numerical/actual-1))),
                     passed=ok)
        result["cases"].append(case)
        passed &= ok
        print(f"Q={p['Q']}: JeansPy/JAM {error:.3g}; public analytic/kernel "
              f"{fine['public_analytic_los']['max_relative_difference']:.3g}; "
              f"numerical LOS refinement {numerical_errors}; pass={ok}", flush=True)
    result.update(passed=passed, elapsed_seconds=time.perf_counter()-start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    if not passed:
        raise SystemExit("Axisymmetric JAM benchmark did not meet its declared tolerances; see the JSON report")


if __name__ == "__main__":
    main()
