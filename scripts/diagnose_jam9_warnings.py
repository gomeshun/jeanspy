"""Locate preserved JAM warnings using the recorded, bounded follow-up plan."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jeanspy-jam9-mpl")
import numpy as np
from jeanspy.axisymmetric import G, PlummerTracer
from jampy.axi.jam_axi_intr import jam_axi_intr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol_path = ROOT/"validation/release/jam9_protocol.json"
    p = json.loads(protocol_path.read_text())
    if version("jampy") != p["jam_version"]:
        raise ValueError("Use the frozen JAM version")
    a, mass = p["physical"]["a_pc"], p["physical"]["mass_Msun"]
    R, z = [np.asarray(p["intrinsic_coordinates_pc"][k]) for k in ("R", "z")]
    tracer = PlummerTracer(a, 1.)
    def potential(R, z):
        return -G*mass/np.sqrt(a*a+R*R+z*z)
    def dpotential(R, z):
        coefficient = G*mass/(a*a+R*R+z*z)**1.5
        return coefficient*R, coefficient*z
    def dtracer(R, z):
        return -5*(R*R+z*z)/(a*a+R*R+z*z), np.zeros_like(R+z)
    settings = p["jam_intrinsic"]
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        jampy_version=version("jampy"), runs=[])
    configurations = [(r, t, True) for r, t in settings["resolutions"]]
    configurations.append((*settings["resolutions"][-1], False))
    for nrad, nang, spectral in configurations:
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = jam_axi_intr((tracer.density, dtracer), (potential, dpotential),
                0., R, z, beta=0., nrad=nrad, nang=nang,
                rmin=settings["rmin_pc"], rmax=settings["rmax_pc"],
                spectral_derivs=spectral, ml=1., plot=False, quiet=True)
        report["runs"].append(dict(nrad=nrad, nang=nang, spectral_derivs=spectral,
            wall_seconds=time.perf_counter()-start, moments=np.asarray(result.model).tolist(),
            returned_moments_finite=bool(np.isfinite(result.model).all()), warnings=[
                dict(message=str(w.message), filename=Path(w.filename).name, line=w.lineno,
                     source_sha256=hashlib.sha256(Path(w.filename).read_bytes()).hexdigest()) for w in caught]))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps([dict(nrad=r["nrad"], nang=r["nang"], spectral=r["spectral_derivs"],
        warnings=r["warnings"], finite=r["returned_moments_finite"]) for r in report["runs"]], indent=2))


if __name__ == "__main__":
    main()
