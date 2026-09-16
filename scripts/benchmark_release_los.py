#!/usr/bin/env python3
"""Run the frozen matched LOS accuracy/time protocol, one bounded role at a time.

For the independent reference use a=GM=1, A=1+R^2, T=A+z^2,
C=3/(4*pi*q), w=(1-q^2)*A/T. Direct integration of the vertical Jeans
equation gives p=nu*sigma_z^2=C*q^5/(6*T^3)*2F1(5/2,3;4;w).
Differentiate this expression analytically at fixed z. With beta_z=0,
nu*vphi^2=p+R*dp/dR+R*nu*dPhi/dR. The LOS integrand is therefore
p+sin(i)^2*x^2*((dp/dR)/R+nu/T^(3/2)). Integrate it over the full LOS
and divide by the exact projected Plummer density. Restore units with GM/a.
This reference uses no JeansPy density, force, pressure or quadrature helper.
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
import subprocess
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jeanspy-los-mpl")
IMPORT_START = time.perf_counter()
import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1
from jeanspy.axisymmetric import G


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(safe(data), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def measure(fn, synchronize=lambda x: x):
    start = time.perf_counter()
    value = synchronize(fn())
    return time.perf_counter() - start, value


def coordinates(q, count, protocol):
    c, inc = protocol["coordinates"], protocol["physical"]["inclination_rad"]
    r = np.geomspace(c["radius_min_pc"], c["radius_max_pc"], count)
    phi = 2*np.pi*((np.arange(count)*c["azimuth_step"]) % 1)
    qp = np.sqrt(np.cos(inc)**2 + q*q*np.sin(inc)**2)
    return r*np.cos(phi), qp*r*np.sin(phi)


def pressure(R, z, q):
    A = 1 + R*R
    T = A + z*z
    w = (1-q*q)*A/T
    C = 3/(4*np.pi*q)
    F = hyp2f1(2.5, 3, 4, w)
    coefficient = C*q**5/6
    p = coefficient*F/T**3
    # dp/dR divided by R: well-defined also at R=0.
    derivative = coefficient*(-6*F/T**4 + (15/8)*hyp2f1(3.5, 4, 5, w)
        *2*(1-q*q)*z*z/T**5)
    return p, derivative


def projected_reference(x_pc, y_pc, q, protocol, level):
    p, cfg = protocol["physical"], protocol["reference"]
    x, y = x_pc/p["a_pc"], y_pc/p["a_pc"]
    si, ci = np.sin(p["inclination_rad"]), np.cos(p["inclination_rad"])
    qp = np.sqrt(ci*ci + q*q*si*si)
    surface = (1+x*x+(y/qp)**2)**-2/(np.pi*qp)
    def integrand(ell):
        Y, z = y*ci+ell*si, -y*si+ell*ci
        R = np.hypot(x, Y)
        T = 1+R*R+z*z
        nu = 3/(4*np.pi*q)*(1+R*R+(z/q)**2)**-2.5
        press, derivative = pressure(R, z, q)
        return (press + si*si*x*x*(derivative+nu/T**1.5))/surface
    value, error = quad(integrand, -np.inf, np.inf, epsabs=cfg["epsabs"][level],
        epsrel=cfg["epsrel"][level], limit=cfg["quad_limit"])
    scale = G*p["mass_Msun"]/p["a_pc"]
    return value*scale, error*scale


def accuracy(value, reference):
    value, reference = np.asarray(value, float), np.asarray(reference, float)
    finite = bool(np.isfinite(value).all() and (value > 0).all()
        and np.isfinite(reference).all() and (reference > 0).all())
    if not finite:
        return dict(finite_positive=False, max_relative_second_moment=None,
            rms_relative_second_moment=None, max_relative_sigma=None)
    residual = value/reference - 1
    return dict(finite_positive=True, signed_relative_second_moment=residual,
        max_relative_second_moment=float(np.max(np.abs(residual))),
        rms_relative_second_moment=float(np.sqrt(np.mean(residual**2))),
        max_relative_sigma=float(np.max(np.abs(np.sqrt(value/reference)-1))))


def reference_role(protocol, report, persist):
    cfg, physical = protocol["reference"], protocol["physical"]
    checks = []
    report["intrinsic_checks"] = checks
    for q in physical["tracer_q"]:
        for R, z in cfg["intrinsic_check_points_in_a"]:
            C = 3/(4*np.pi*q)
            def integrand(t):
                S, T = 1+R*R+(t/q)**2, 1+R*R+t*t
                return C*t*S**-2.5*T**-1.5
            pnum = quad(integrand, abs(z), np.inf, epsabs=2e-13, epsrel=2e-12)[0]
            dnum = quad(lambda t: integrand(t)*(-5/(1+R*R+(t/q)**2)
                -3/(1+R*R+t*t)), abs(z), np.inf, epsabs=2e-13, epsrel=2e-12)[0]
            exact = np.asarray(pressure(R, z, q))
            errors = np.abs(exact/np.asarray([pnum, dnum])-1)
            checks.append(dict(q=q, R=R, z=z, hypergeometric=exact,
                direct_integral=[pnum, dnum], relative_error=errors,
                passed=bool(np.max(errors) <= cfg["max_relative_intrinsic_integral_check"])))
    persist()
    for q in physical["tracer_q"]:
        for count in protocol["coordinates"]["counts"]:
            x, y = coordinates(q, count, protocol)
            values = [np.array([projected_reference(xx, yy, q, protocol, level)
                for xx, yy in zip(x, y)]) for level in (0, 1)]
            refined = values[1][:, 0]
            row = dict(q=q, n_positions=count, x_pc=x, y_pc=y,
                quadrature=values, refinement=accuracy(values[0][:, 0], refined),
                sigma_los2=refined)
            row["passed"] = row["refinement"]["finite_positive"] and (
                row["refinement"]["max_relative_second_moment"] <= cfg["max_relative_refinement"])
            if q == 1:
                exact = 3*np.pi*G*physical["mass_Msun"]/(64*np.sqrt(physical["a_pc"]**2+x*x+y*y))
                row["analytic_check"] = accuracy(refined, exact)
                row["passed"] &= row["analytic_check"]["finite_positive"] and (
                    row["analytic_check"]["max_relative_second_moment"] <= cfg["max_relative_spherical_analytic_check"])
                row["sigma_los2"] = exact
            row["status"] = "completed" if row["passed"] else "failed_reference_gate"
            report["rows"].append(row)
            persist()
    report["status"] = "completed" if all(c["passed"] for c in checks + report["rows"]) else "failed_reference_gate"


def physical_params(q, protocol, spherical=False):
    p = protocol["physical"]
    result = dict(re_pc=p["a_pc"], rs_pc=p["a_pc"],
        rhos_Msunpc3=3*p["mass_Msun"]/(4*np.pi*p["a_pc"]**3), r_t_pc=np.inf)
    if spherical:
        result.update(a=p["alpha"], b=p["beta"], g=p["gamma"], beta_ani=0.)
    else:
        result.update(q=q, Q=p["halo_Q"], alpha=p["alpha"], beta=p["beta"],
            gamma=p["gamma"], beta_z=p["beta_z"], inclination=p["inclination_rad"])
    return result


def make_jeanspy(role, spherical, order, q, x, y, protocol, row):
    is_jax = role.startswith("jax")
    if is_jax:
        import jax
        from jeanspy import model_numpyro as module
        from jeanspy.axisymmetric_numpyro import AxisymmetricDSphModel
        sync = jax.block_until_ready
        jax.clear_caches()
    else:
        from jeanspy import model as module
        from jeanspy.axisymmetric import AxisymmetricDSphModel
        sync = lambda value: value
    started = time.perf_counter()
    params = physical_params(q, protocol, spherical)
    if spherical:
        model = module.DSphModel(submodels={"StellarModel": module.PlummerModel(),
            "DMModel": module.ZhaoModel(), "AnisotropyModel": module.ConstantAnisotropyModel()})
    else:
        model = AxisymmetricDSphModel(order, order, order)
    row["model_construction_seconds"] = time.perf_counter()-started
    cfg = protocol["spherical_settings"]
    if is_jax:
        row["input_transfer_seconds"], data = measure(lambda: jax.device_put(
            (np.asarray(x, np.float64), np.asarray(y, np.float64), params)), sync)
        xx, yy, parameters = data
        if spherical:
            def forward(xx, yy, pars):
                return model.sigmalos2((xx*xx+yy*yy)**.5, params=pars, backend="kernel",
                    n_u=order, n_kernel=cfg["n_kernel"], u_max=cfg["u_max"],
                    kernel_outer_transform=cfg["kernel_outer_transform"], constant_kernel_backend="jax",
                    dm_mass_method=cfg["dm_mass_method"], dm_mass_n_steps=cfg["dm_mass_n_steps"])
        else:
            def forward(xx, yy, pars):
                return model.sigmalos2(xx, yy, params=pars)
        compiled = jax.jit(forward)
        # Prepare all device inputs outside timing, with no host scalar promotion.
        factors = [1.] + (1+.01*np.sin(np.arange(protocol["timing"]["warm_repeats"])*.71)).tolist()
        started = time.perf_counter()
        arguments = [dict(parameters, rhos_Msunpc3=jax.device_put(np.asarray(
            params["rhos_Msunpc3"]*factor, np.float64))) for factor in factors]
        sync(arguments)
        row["mass_sequence_input_preparation_seconds"] = time.perf_counter()-started
        return lambda index: compiled(xx, yy, arguments[index]), sync
    factors = [1.] + (1+.01*np.sin(np.arange(protocol["timing"]["warm_repeats"])*.71)).tolist()
    arguments = [dict(params, rhos_Msunpc3=params["rhos_Msunpc3"]*factor) for factor in factors]
    if spherical:
        def call(index):
            model.update(**arguments[index])
            return model.sigmalos2(np.hypot(x, y), n=order, n_kernel=cfg["n_kernel"], ignore_RuntimeWarning=False)
    else:
        def call(index):
            return model.sigmalos2(x, y, params=arguments[index])
    return call, sync


def timing_role(role, protocol, reference, report, persist):
    is_jax = role.startswith("jax")
    if role == "jam":
        from validate_axisymmetric_jam9 import fit_tracer
        from jeanspy.axisymmetric import AxisymmetricPlummerModel
        from jampy.axi.jam_axi_proj import jam_axi_proj
        report["third_party_source_sha256"] = {
            "jam_axi_proj": digest(inspect.getsourcefile(jam_axi_proj)),
            "jam_axi_intr.py": digest(Path(inspect.getsourcefile(jam_axi_proj)).with_name("jam_axi_intr.py"))}
    factors = 1+.01*np.sin(np.arange(protocol["timing"]["warm_repeats"])*.71)
    report["mass_factors"] = factors
    fits = {}
    report["mge_fits"] = []
    for ref in reference["rows"]:
        q, count = ref["q"], ref["n_positions"]
        prep, xy = measure(lambda: coordinates(q, count, protocol))
        x, y = xy
        expected = np.asarray(ref["sigma_los2"])
        if not np.array_equal(x, ref["x_pc"]) or not np.array_equal(y, ref["y_pc"]):
            raise ValueError("reference coordinates differ")
        solvers = ["jam"] if role == "jam" else ["axisymmetric"] + (["spherical"] if q == 1 else [])
        for solver in solvers:
            settings = (protocol["jam_projection"] if role == "jam" else
                protocol["spherical_jax_orders" if is_jax else "spherical_classical_orders"]
                if solver == "spherical" else protocol["axisymmetric_orders"])
            for tier, setting in enumerate(settings):
                row = dict(q=q, n_positions=count, solver=solver, tier=tier,
                    setting=setting, status="running", coordinate_preparation_seconds=prep)
                report["rows"].append(row)
                persist()
                with warnings.catch_warnings(record=True) as caught, redirect_stdout(io.StringIO()) as captured:
                    warnings.simplefilter("always")
                    try:
                        if role == "jam":
                            key = (q, tier)
                            p = protocol["physical"]
                            if key not in fits:
                                tracer = AxisymmetricPlummerModel(p["a_pc"], q)
                                fit_time, fit = measure(lambda: fit_tracer(tracer, setting["ngauss"], protocol["mge_fit"]))
                                weights, widths, metadata = fit
                                qp = tracer.projected_axis_ratio(p["inclination_rad"])
                                fits[key] = (weights*q/qp, widths, np.full(widths.size, qp))
                                report["mge_fits"].append(dict(q=q, tier=tier, seconds=fit_time, **metadata))
                                persist()
                            row["mge_fit_key"] = [q, tier]
                            sequence = [1.] + factors.tolist()
                            def call(index):
                                def potential(R, z):
                                    return -G*p["mass_Msun"]*sequence[index]/np.sqrt(p["a_pc"]**2+R*R+z*z)
                                result = jam_axi_proj(fits[key], potential, np.rad2deg(p["inclination_rad"]),
                                    0., .648/np.pi, x, y, beta=0., ml=1., sigmapsf=0., pixsize=0.,
                                    interp=False, nang=setting["nang"], nlos=setting["nlos"],
                                    spectral_derivs=True, moment="zz", plot=False, quiet=True)
                                return np.square(result.model)
                            sync = lambda value: value
                        else:
                            call, sync = make_jeanspy(role, solver == "spherical", setting, q, x, y, protocol, row)
                        persist()
                        row["first_prediction_seconds"], first = measure(lambda: call(0), sync)
                        row["first_result_transfer_seconds"], first_host = measure(lambda: np.asarray(first))
                        row["sigma_los2"] = first_host
                        row["accuracy"] = accuracy(first_host, expected)
                        row["repetitions"] = []
                        persist()
                        for index, factor in enumerate(factors, 1):
                            duration, value = measure(lambda: call(index), sync)
                            transfer, host = measure(lambda: np.asarray(value))
                            row["repetitions"].append(dict(mass_factor=factor, seconds=duration,
                                result_transfer_seconds=transfer, accuracy=accuracy(host, expected*factor)))
                            persist()
                        durations = [v["seconds"] for v in row["repetitions"]]
                        row["warm"] = dict(median_seconds=float(np.median(durations)),
                            p10_seconds=float(np.quantile(durations, .1)), p90_seconds=float(np.quantile(durations, .9)))
                        checks = [row["accuracy"]] + [r["accuracy"] for r in row["repetitions"]]
                        row["passed"] = all(c["finite_positive"] and c["max_relative_second_moment"] <=
                            protocol["accuracy"]["primary_second_moment_relative_limit"] for c in checks)
                        row["status"] = "completed_passed_accuracy" if row["passed"] else "completed_failed_accuracy"
                    except Exception:
                        row.update(status="execution_failed", passed=False, exception=traceback.format_exc())
                    finally:
                        row["warnings"] = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
                        row["stdout"] = captured.getvalue()
                        persist()
                print(json.dumps({k: row[k] for k in ("q", "n_positions", "solver", "tier", "status")}), flush=True)
    report["status"] = "completed" if all(r["status"].startswith("completed") for r in report["rows"]) else "completed_with_execution_failures"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=["reference", "numpy", "jax-cpu", "jax-gpu", "jam"])
    parser.add_argument("--protocol", type=Path, default=ROOT/"validation/release/los_benchmark_protocol.json")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path; retain earlier attempts")
    protocol = json.loads(args.protocol.read_text())
    if sorted(os.sched_getaffinity(0)) != protocol["resources"]["cpu_affinity"]:
        parser.error("run under taskset -c 0-7")
    if any(os.environ.get(k) != "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")):
        parser.error("set BLAS/OpenMP thread counts before starting Python")
    packages = ["numpy", "scipy"]
    report = dict(schema_version=1, role=args.role, status="running", protocol=protocol,
        protocol_sha256=digest(args.protocol), script_sha256=digest(__file__),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        runtime_sources={str(f.relative_to(ROOT)): digest(f) for f in sorted((ROOT/"src/jeanspy").rglob("*.py"))},
        helper_sha256={"validate_axisymmetric_jam9.py": digest(ROOT/"scripts/validate_axisymmetric_jam9.py")},
        cpu_affinity=sorted(os.sched_getaffinity(0)), platform=platform.platform(), python=platform.python_version(),
        cpu_model=next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
        G_pc_kms2_per_Msun=G, started_utc=datetime.now(timezone.utc).isoformat(), rows=[])
    if args.role.startswith("jax"):
        from jeanspy._jax_env import configure_jax_environment
        configure_jax_environment()
        import jax
        from jeanspy import model_numpyro
        from jeanspy import axisymmetric_numpyro
        if jax.default_backend() != args.role.split("-")[1] or not jax.config.jax_enable_x64:
            parser.error("use the declared JAX device with float64 enabled before imports")
        report["devices"] = [str(device) for device in jax.devices()]
        packages += ["jax", "jaxlib"]
    if args.role == "numpy":
        from jeanspy import model
    if args.role == "jam":
        for name, key in [("jampy", "jam_version"), ("mgefit", "mgefit_version")]:
            if version(name) != protocol[key]:
                parser.error(f"protocol requires {name}=={protocol[key]}")
        import mgefit
        from jampy.axi.jam_axi_proj import jam_axi_proj
        packages += ["jampy", "mgefit"]
    report["versions"] = {name: version(name) for name in packages}
    report["imports_and_device_init_seconds"] = time.perf_counter()-IMPORT_START
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def persist():
        report["elapsed_seconds"] = time.perf_counter()-started
        report["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        save(args.output, report)
    persist()
    try:
        if args.role == "reference":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reference_role(protocol, report, persist)
                report["warnings"] = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
        else:
            if args.reference is None:
                raise ValueError("a completed independent reference is required")
            reference = json.loads(args.reference.read_text())
            if reference["status"] != "completed" or reference["protocol_sha256"] != report["protocol_sha256"]:
                raise ValueError("reference failed or used a different protocol")
            report["reference_sha256"] = digest(args.reference)
            timing_role(args.role, protocol, reference, report, persist)
    except Exception:
        report.update(status="execution_failed", exception=traceback.format_exc())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    persist()
    print(json.dumps(dict(role=args.role, status=report["status"], elapsed_seconds=report["elapsed_seconds"])), flush=True)
    return int(report["status"] not in ("completed",))


if __name__ == "__main__":
    raise SystemExit(main())
