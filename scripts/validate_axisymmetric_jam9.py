#!/usr/bin/env python3
"""Execute the frozen Plummer/isotropic accuracy protocol with JAM 9.0.2.

Install jampy==9.0.2 and mgefit==6.2.6 in a separate environment; the existing
benchmark extra intentionally retains the legacy cylindrical JAM 8 API.
This script imports the repository source and never changes third-party code.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import io
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jeanspy-jam9-mpl")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from jeanspy.axisymmetric import AxisymmetricDSphModel, G, PlummerTracer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all() or np.any(expected <= 0):
        return None
    return float(np.max(np.abs(actual/expected - 1.)))


def atomic_save(path, report):
    def json_safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)  # Preserve nonfinite failures in valid JSON.
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def fit_tracer(tracer, count, settings):
    import mgefit
    radius = np.geomspace(settings["rmin_pc"], settings["rmax_pc"], settings["samples"])
    density = tracer.density(radius, 0.)
    amplitude, scale = float(density.max()), tracer.a_pc
    result = mgefit.fit_1d(radius/scale, density/amplitude, ngauss=count,
        inner_slope=settings["inner_slope"], outer_slope=settings["outer_slope"],
        quiet=True, plot=False)
    integral, sigma = np.asarray(result.sol) * np.array([[amplitude*scale], [scale]])
    peaks = integral/(np.sqrt(2*np.pi)*sigma)
    prediction = np.exp(-.5*(radius[:, None]/sigma)**2) @ peaks
    lo, hi = settings["assessment_range_pc"]
    mask = (radius >= lo) & (radius <= hi)
    return integral, sigma, dict(intrinsic_density_integrals=integral.tolist(),
        sigma_pc=sigma.tolist(), requested_components=count, actual_components=len(sigma),
        density_relative_error=relative(prediction[mask], density[mask]),
        assessment_range_pc=[lo, hi])


def run_case(q, protocol, case, save):
    from jampy.axi.jam_axi_intr import jam_axi_intr
    from jampy.axi.jam_axi_proj import jam_axi_proj
    p = protocol["physical"]
    a, mass, inc = p["a_pc"], p["mass_Msun"], p["inclination_rad"]
    tracer = PlummerTracer(a, q)
    params = dict(re_pc=a, rs_pc=a, rhos_Msunpc3=3*mass/(4*np.pi*a**3),
        q=q, Q=p["halo_Q"], alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"],
        beta_z=p["beta_z"], inclination=inc)
    R, z = [np.asarray(protocol["intrinsic_coordinates_pc"][k]) for k in ("R", "z")]
    x, y = [np.asarray(protocol["sky_coordinates_pc"][k]) for k in ("x", "y")]

    def potential(R, z):
        return -G*mass/np.sqrt(a*a + R*R + z*z)

    def potential_gradient(R, z):
        coefficient = G*mass/(a*a + R*R + z*z)**1.5
        return coefficient*R, coefficient*z

    def tracer_gradient(R, z):
        denominator = a*a + R*R + z*z/(q*q)
        return (-5*(R*R + z*z/(q*q))/denominator,
                -5*R*z*(1 - 1/(q*q))/denominator)

    case.update(params=params, jeanspy=[], jam_intrinsic=[], jam_projected=[])
    for n in protocol["jeanspy_orders"]:
        started = time.perf_counter()
        model = AxisymmetricDSphModel(n, n, n)
        moments = np.asarray(model.intrinsic_moments(R, z, params=params))
        projected = model.sigmalos2(x, y, params=params)
        case["jeanspy"].append(dict(n=n, intrinsic=moments.tolist(),
            projected=np.asarray(projected).tolist(), wall_seconds=time.perf_counter()-started))
        save()

    settings = protocol["jam_intrinsic"]
    for nrad, nang in settings["resolutions"]:
        started = time.perf_counter()
        result = jam_axi_intr(tracer.density, potential, 0., R, z, beta=0.,
            nrad=nrad, nang=nang, rmin=settings["rmin_pc"], rmax=settings["rmax_pc"],
            spectral_derivs=True, ml=1., plot=False, quiet=True)
        case["jam_intrinsic"].append(dict(nrad=nrad, nang=nang,
            spectral_derivs=True, intrinsic=np.asarray(result.model[[0, 1, 3]]).tolist(),
            wall_seconds=time.perf_counter()-started))
        save()
    nrad, nang = settings["resolutions"][-1]
    started = time.perf_counter()
    analytic = jam_axi_intr((tracer.density, tracer_gradient), (potential, potential_gradient),
        0., R, z, beta=0., nrad=nrad, nang=nang, rmin=settings["rmin_pc"],
        rmax=settings["rmax_pc"], spectral_derivs=False, ml=1., plot=False, quiet=True)
    case["jam_analytic_derivative_control"] = dict(
        intrinsic=np.asarray(analytic.model[[0, 1, 3]]).tolist(),
        wall_seconds=time.perf_counter()-started)
    save()

    for configuration in protocol["jam_projection"]["resolutions"]:
        started = time.perf_counter()
        weights, widths, fit = fit_tracer(tracer, configuration["ngauss"], protocol["mge_fit"])
        fit_seconds = time.perf_counter()-started
        qproj = float(tracer.projected_axis_ratio(inc))
        surface = weights*q/qproj
        projected_mge = (surface, widths, np.full(widths.size, qproj))
        started = time.perf_counter()
        # distance=.648/pi Mpc makes one arcsec equal one pc. The callable
        # potential already contains JeansPy's G; no JAM G rescaling applies.
        output = jam_axi_proj(projected_mge, potential, np.rad2deg(inc), 0.,
            .648/np.pi, x, y, beta=0., ml=1., sigmapsf=0., pixsize=0.,
            interp=False, nang=configuration["nang"], nlos=configuration["nlos"],
            spectral_derivs=True, moment="zz", plot=False, quiet=True)
        case["jam_projected"].append(dict(**configuration, mge=fit, fit_seconds=fit_seconds,
            projected=np.square(output.model).tolist(), wall_seconds=time.perf_counter()-started))
        save()

    jp, ji, jl = case["jeanspy"], case["jam_intrinsic"], case["jam_projected"]
    errors = dict(
        intrinsic_cross_code=relative(jp[-1]["intrinsic"], ji[-1]["intrinsic"]),
        projected_cross_code=relative(jp[-1]["projected"], jl[-1]["projected"]),
        jeanspy_intrinsic_refinement=relative(jp[-2]["intrinsic"], jp[-1]["intrinsic"]),
        jeanspy_projected_refinement=relative(jp[-2]["projected"], jp[-1]["projected"]),
        jam_intrinsic_refinement=relative(ji[-2]["intrinsic"], ji[-1]["intrinsic"]),
        jam_projected_refinement=relative(jl[-2]["projected"], jl[-1]["projected"]),
        mge_density=jl[-1]["mge"]["density_relative_error"],
        spectral_vs_analytic_derivatives=relative(ji[-1]["intrinsic"],
            case["jam_analytic_derivative_control"]["intrinsic"]),
    )
    thresholds = protocol["acceptance"]
    limits = dict(intrinsic_cross_code="finest_cross_code_relative",
        projected_cross_code="finest_cross_code_relative",
        jeanspy_intrinsic_refinement="jeanspy_refinement_relative",
        jeanspy_projected_refinement="jeanspy_refinement_relative",
        jam_intrinsic_refinement="jam_refinement_relative",
        jam_projected_refinement="jam_refinement_relative", mge_density="mge_density_relative")
    if q == 1:
        intrinsic = np.broadcast_to(G*mass/(6*np.sqrt(a*a+R*R+z*z)), (3, R.size))
        projected = 3*np.pi*G*mass/(64*np.sqrt(a*a+x*x+y*y))
        case["analytic"] = dict(intrinsic=intrinsic.tolist(), projected=projected.tolist())
        for name, values in [("jeanspy", jp[-1]), ("jam", {**ji[-1], **jl[-1]})]:
            for kind, reference in [("intrinsic", intrinsic), ("projected", projected)]:
                errors[f"{name}_{kind}_analytic"] = relative(values[kind], reference)
                limits[f"{name}_{kind}_analytic"] = f"{name}_analytic_relative"
    case["errors"] = errors
    case["gates"] = {key: dict(value=errors[key], threshold=thresholds[limit],
        passed=errors[key] is not None and errors[key] <= thresholds[limit]) for key, limit in limits.items()}
    case["passed"] = all(gate["passed"] for gate in case["gates"].values())
    case["status"] = "passed" if case["passed"] else "failed_accuracy_gate"
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT/"validation/release/jam9_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output path; prior runs must remain intact")
    protocol = json.loads(args.protocol.read_text())
    for package in ("jampy", "mgefit"):
        if version(package) != protocol[f"{'jam' if package == 'jampy' else package}_version"]:
            raise ValueError(f"Protocol requires the recorded {package} version")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1" or os.environ.get("OMP_NUM_THREADS") != "1":
        raise ValueError("Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before Python starts")
    from jampy.axi.jam_axi_intr import jam_axi_intr
    from jampy.axi.jam_axi_proj import jam_axi_proj
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=protocol, protocol_sha256=digest(args.protocol),
        script_sha256=digest(__file__), started_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), python=platform.python_version(),
        versions={name: version(name) for name in ("jampy", "mgefit", "numpy", "scipy")},
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in
            [ROOT/"src/jeanspy/axisymmetric.py", ROOT/"src/jeanspy/_axisymmetric_params.py"]},
        jam_source_sha256={obj.__name__: digest(inspect.getsourcefile(obj)) for obj in (jam_axi_intr, jam_axi_proj)},
        G_pc_kms2_per_Msun=G, status="running", cases=[])
    started = time.perf_counter()
    def save():
        report["elapsed_seconds"] = time.perf_counter()-started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        atomic_save(args.output, report)
    save()
    with warnings.catch_warnings(record=True) as recorded, redirect_stdout(io.StringIO()) as log:
        warnings.simplefilter("always")
        for q in protocol["physical"]["tracer_q"]:
            case = dict(q=q, status="running")
            report["cases"].append(case)
            try:
                run_case(q, protocol, case, save)
            except Exception:
                case.update(status="exception", passed=False, exception=traceback.format_exc())
                save()
        report["warnings"] = [dict(category=w.category.__name__, message=str(w.message)) for w in recorded]
        report["stdout"] = log.getvalue()
    report["status"] = "passed" if all(c.get("passed", False) for c in report["cases"]) else "failed"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(dict(status=report["status"], elapsed_seconds=report["elapsed_seconds"],
        cases=[dict(q=c["q"], status=c["status"], errors=c.get("errors"), exception=c.get("exception"))
               for c in report["cases"]]), indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
