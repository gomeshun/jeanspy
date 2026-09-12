"""Locate preserved JAM warnings using the recorded, bounded follow-up plan.

The Linux parent enforces a wall-clock deadline. Its worker sets an address-
space bound before importing numerical libraries and saves each configuration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
WALL_SECONDS = 120
ADDRESS_SPACE_BYTES = 16 * 1024**3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(path, report):
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_single_thread():
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1 before starting Python")


def apply_address_space_bound():
    import resource
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    bound = ADDRESS_SPACE_BYTES if hard == resource.RLIM_INFINITY else min(ADDRESS_SPACE_BYTES, hard)
    resource.setrlimit(resource.RLIMIT_AS, (bound, bound))
    return bound


def warning_record(warning):
    record = dict(message=str(warning.message), category=warning.category.__name__,
                  filename=Path(warning.filename).name, line=warning.lineno)
    try:
        record["source_sha256"] = digest(warning.filename)
    except OSError:
        record.update(source_sha256=None, source_read_exception=traceback.format_exc())
    return record


def run_configuration(report, save, configuration, call):
    """Retain output and warnings even when a call or source-hash read fails."""
    run = dict(**configuration, status="running", warnings=[])
    report["runs"].append(run)
    save()
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            moments, finite = call()
            run.update(status="completed", moments=moments, returned_moments_finite=finite)
        except Exception:
            run.update(status="exception", exception=traceback.format_exc())
    run["warnings"] = [warning_record(warning) for warning in caught]
    if run["status"] == "completed" and any("source_read_exception" in w for w in run["warnings"]):
        run["status"] = "completed_with_source_errors"
    run["wall_seconds"] = time.perf_counter() - started
    save()


def worker(output):
    report = json.loads(output.read_text())
    if (report.get("status") != "running"
            or report.get("supervisor_pid") != os.getppid()
            or report.get("script_sha256") != digest(__file__)):
        raise ValueError("The internal worker requires its active supervisor's new report")
    started = time.perf_counter()
    def save():
        report["worker_elapsed_seconds"] = time.perf_counter() - started
        atomic_save(output, report)
    try:
        require_single_thread()
        report["enforced_address_space_bytes"] = apply_address_space_bound()
        save()
        # Thread and memory limits are established before NumPy/JAM imports.
        sys.path.insert(0, str(ROOT / "src"))
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/jeanspy-jam9-mpl")
        os.environ.setdefault("MPLBACKEND", "Agg")
        import numpy as np
        from jeanspy.axisymmetric import G, PlummerTracer
        from jampy.axi.jam_axi_intr import jam_axi_intr
        protocol = json.loads((ROOT / "validation/release/jam9_protocol.json").read_text())
        report["jampy_version"] = version("jampy")
        if report["jampy_version"] != protocol["jam_version"]:
            raise ValueError("Use the frozen JAM version")
        a, mass = protocol["physical"]["a_pc"], protocol["physical"]["mass_Msun"]
        R, z = [np.asarray(protocol["intrinsic_coordinates_pc"][key]) for key in ("R", "z")]
        tracer = PlummerTracer(a, 1.)
        def potential(R, z):
            return -G * mass / np.sqrt(a*a + R*R + z*z)
        def dpotential(R, z):
            coefficient = G * mass / (a*a + R*R + z*z)**1.5
            return coefficient*R, coefficient*z
        def dtracer(R, z):
            return -5 * (R*R + z*z) / (a*a + R*R + z*z), np.zeros_like(R+z)
        settings = protocol["jam_intrinsic"]
        configurations = [(r, t, True) for r, t in settings["resolutions"]]
        configurations.append((*settings["resolutions"][-1], False))
        for nrad, nang, spectral in configurations:
            def call():
                result = jam_axi_intr((tracer.density, dtracer), (potential, dpotential),
                    0., R, z, beta=0., nrad=nrad, nang=nang,
                    rmin=settings["rmin_pc"], rmax=settings["rmax_pc"],
                    spectral_derivs=spectral, ml=1., plot=False, quiet=True)
                return np.asarray(result.model).tolist(), bool(np.isfinite(result.model).all())
            run_configuration(report, save,
                dict(nrad=nrad, nang=nang, spectral_derivs=spectral), call)
        report["status"] = ("completed" if all(r["status"] == "completed" for r in report["runs"])
                            else "completed_with_errors")
    except Exception:
        report.update(status="exception", exception=traceback.format_exc())
    save()
    return 0 if report["status"] == "completed" else 1


def supervise(command, output, wall_seconds):
    """Bound the worker independently of Python callbacks or native code."""
    started = time.perf_counter()
    timed_out = False
    with subprocess.Popen(command) as process:
        try:
            code = process.wait(timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    report = json.loads(output.read_text())
    report.update(worker_exit_code=code, elapsed_seconds=time.perf_counter()-started,
                  completed_utc=datetime.now(timezone.utc).isoformat())
    if timed_out or report["status"] == "running":
        report["status"] = "stopped_time_limit" if timed_out else "worker_exited_without_completion"
        for run in report["runs"]:
            if run["status"] == "running":
                run["status"] = report["status"]
    atomic_save(output, report)
    return 0 if code == 0 and report["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        supervisor_pid=os.getpid(),
        protocol_sha256=digest(ROOT / "validation/release/jam9_protocol.json"),
        diagnostic_plan_sha256=digest(ROOT / "validation/release/jam9_warning_diagnostic.md"),
        script_sha256=digest(__file__), status="running", runs=[],
        limits=dict(wall_seconds=WALL_SECONDS, cpu_threads=1, address_space_bytes=ADDRESS_SPACE_BYTES),
        thread_environment={name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")})
    atomic_save(args.output, report)
    try:
        require_single_thread()
        return supervise([sys.executable, str(Path(__file__).resolve()), "--worker",
                          "--output", str(args.output.resolve())], args.output, WALL_SECONDS)
    except Exception:
        report = json.loads(args.output.read_text())
        report.update(status="exception", exception=traceback.format_exc(),
                      completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_save(args.output, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
