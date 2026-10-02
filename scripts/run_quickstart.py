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
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLE_TIMEOUT = 1800


def run_example(backend, directory, environment, *, source_root=ROOT,
                example_timeout=DEFAULT_EXAMPLE_TIMEOUT):
    print(f"Executing {backend} (timeout: {example_timeout}s)", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, str(source_root / f"examples/docs_quickstart_{backend}.py"),
             "--output-dir", str(directory / backend)],
            cwd=source_root, env=environment, text=True, capture_output=True,
            timeout=example_timeout,
        )
    except subprocess.TimeoutExpired as error:
        # TimeoutExpired can carry bytes even when text=True. Keep partial logs
        # in the build directory rather than losing the expensive run's evidence.
        for stream, value in (("stdout", error.stdout), ("stderr", error.stderr)):
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            (directory / f"{backend}.{stream}.txt").write_text(value or "")
        raise RuntimeError(
            f"{backend} example timed out after {example_timeout}s; "
            f"partial stdout/stderr retained in {directory}"
        ) from error
    (directory / f"{backend}.stdout.txt").write_text(result.stdout)
    (directory / f"{backend}.stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{backend} example failed:\n{result.stderr}")
    print(f"Completed {backend}: {directory / backend}", flush=True)
    return backend, result.stdout


def main(build_directory=None, *, source_root=ROOT,
         example_timeout=DEFAULT_EXAMPLE_TIMEOUT, tooling_commit=None):
    if example_timeout <= 0:
        raise ValueError("example timeout must be a positive number of seconds")
    if tooling_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", tooling_commit):
        raise ValueError("tooling commit must be a full 40-character commit SHA")
    source_root = source_root.resolve()
    destination = source_root / "docs/source/_static/quickstart"
    if build_directory is None:
        build_directory = source_root / "docs/_build/quickstart"
    build_directory.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=build_directory.resolve()))
    environment = {**os.environ, "JEANSPY_JAX_PLATFORM": "cpu",
                   "JEANSPY_JAX_ENABLE_X64": "true", "MPLBACKEND": "Agg",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "PYTHONUNBUFFERED": "1"}
    sources = list((source_root / "examples").glob("docs_quickstart_*.py"))
    sources += list((source_root / "src/jeanspy").rglob("*.py"))
    sources += [source_root / "scripts/run_quickstart.py", source_root / "uv.lock"]
    hashes = {p.relative_to(source_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(sources)}
    # Recovery may use newer orchestration without changing released examples,
    # package code or lockfile. Record that helper separately from source hashes.
    runner_path = Path(__file__).resolve()
    runner_hash = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_example, name, directory, environment,
                                   source_root=source_root, example_timeout=example_timeout)
                   for name in ("emcee", "numpyro")]
        results = dict(future.result() for future in futures)
    if any(hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest
           for name, digest in hashes.items()):
        raise RuntimeError("Example source or environment lock changed during execution")
    if hashlib.sha256(runner_path.read_bytes()).hexdigest() != runner_hash:
        raise RuntimeError("Quickstart execution helper changed during execution")
    if (directory / "emcee/observations.csv").read_bytes() != (
        directory / "numpyro/observations.csv"
    ).read_bytes():
        raise RuntimeError("The two examples used different mock observations")
    destination.mkdir(parents=True, exist_ok=True)
    for backend, stdout in results.items():
        (destination / f"{backend}.txt").write_text(stdout)
        for name in ("trace", "autocorrelation", "posterior"):
            shutil.copyfile(directory / backend / f"{name}.png",
                            destination / f"{backend}-{name}.png")
    shutil.copyfile(directory / "emcee/observations.png", destination / "observations.png")
    metadata = {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "emcee", "jax", "numpyro", "arviz", "matplotlib", "corner")},
        "environment": {k: environment[k] for k in
                        ("JEANSPY_JAX_PLATFORM", "JEANSPY_JAX_ENABLE_X64")},
        "source_sha256": hashes,
        "tooling": {"runner_sha256": runner_hash, "commit": tooling_commit,
                    "example_timeout_seconds": example_timeout},
        "data_sha256": hashlib.sha256((directory / "emcee/observations.csv").read_bytes()).hexdigest(),
        "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(destination.iterdir()) if p.suffix in {".png", ".txt"}},
    }
    (destination / "execution.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Published figures and execution results to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path,
                        help="Build directory (default: SOURCE_ROOT/docs/_build/quickstart)")
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="Checked-out source to execute and document")
    parser.add_argument("--example-timeout", type=int, default=DEFAULT_EXAMPLE_TIMEOUT,
                        help="Maximum seconds for each unchanged Quickstart example")
    parser.add_argument("--tooling-commit", help="Full commit SHA containing this execution helper")
    main(**vars(parser.parse_args()))
