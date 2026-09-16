"""Execute both public Quickstart examples and publish their actual outputs.

Raw chains remain in the chosen build directory. Only figures, terminal output
and reproduction metadata are copied into the documentation source tree.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "docs/source/_static/quickstart"


def run_example(backend, directory, environment):
    result = subprocess.run(
        [sys.executable, str(ROOT / f"examples/docs_quickstart_{backend}.py"),
         "--output-dir", str(directory / backend)],
        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=900,
    )
    (directory / f"{backend}.stdout.txt").write_text(result.stdout)
    (directory / f"{backend}.stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{backend} example failed:\n{result.stderr}")
    print(f"Completed {backend}: {directory / backend}", flush=True)
    return backend, result.stdout


def main(build_directory):
    build_directory.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=build_directory.resolve()))
    environment = {**os.environ, "JEANSPY_JAX_PLATFORM": "cpu",
                   "JEANSPY_JAX_ENABLE_X64": "true", "MPLBACKEND": "Agg",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    sources = list((ROOT / "examples").glob("docs_quickstart_*.py"))
    sources += list((ROOT / "src/jeanspy").rglob("*.py"))
    sources += [Path(__file__).resolve(), ROOT / "uv.lock"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(sources)}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_example, name, directory, environment)
                   for name in ("emcee", "numpyro")]
        results = dict(future.result() for future in futures)
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
           for name, digest in hashes.items()):
        raise RuntimeError("Example source or environment lock changed during execution")
    if (directory / "emcee/observations.csv").read_bytes() != (
        directory / "numpyro/observations.csv"
    ).read_bytes():
        raise RuntimeError("The two examples used different mock observations")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for backend, stdout in results.items():
        (DESTINATION / f"{backend}.txt").write_text(stdout)
        for name in ("trace", "autocorrelation", "posterior"):
            shutil.copyfile(directory / backend / f"{name}.png",
                            DESTINATION / f"{backend}-{name}.png")
    shutil.copyfile(directory / "emcee/observations.png", DESTINATION / "observations.png")
    metadata = {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "emcee", "jax", "numpyro", "arviz", "matplotlib", "corner")},
        "environment": {k: environment[k] for k in
                        ("JEANSPY_JAX_PLATFORM", "JEANSPY_JAX_ENABLE_X64")},
        "source_sha256": hashes,
        "data_sha256": hashlib.sha256((directory / "emcee/observations.csv").read_bytes()).hexdigest(),
        "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(DESTINATION.iterdir()) if p.suffix in {".png", ".txt"}},
    }
    (DESTINATION / "execution.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Published figures and execution results to {DESTINATION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, default=ROOT / "docs/_build/quickstart")
    main(parser.parse_args().build_directory)
