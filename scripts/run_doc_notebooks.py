"""Validate or execute the public documentation notebooks in fresh kernels.

Ordinary runs execute the four lightweight tutorials. Use --include-mcmc
explicitly for Quickstart, inference and storage; --write saves reviewed outputs.
No notebook imports a sibling file or relies on the repository as its cwd.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/source"
NOTEBOOKS = ("quickstart", "tutorials/units", "tutorials/backends", "tutorials/models",
             "tutorials/predictions", "tutorials/inference", "tutorials/storage")


def code_hash(notebook):
    source = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    return hashlib.sha256(json.dumps(source).encode()).hexdigest()


def validate(notebook, *, saved=False):
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        compile(cell.source, f"notebook cell {index}", "exec")
        if any(output.output_type == "error" for output in cell.outputs):
            raise ValueError(f"Cell {index} contains an execution error")
        if saved and cell.execution_count is None:
            raise ValueError(f"Cell {index} has no saved execution")
    if saved and notebook.metadata.get("jeanspy", {}).get("execution", {}).get("code_sha256") != code_hash(notebook):
        raise ValueError("Saved outputs do not match the current notebook code")


def execute(path):
    notebook = nbformat.read(path, as_version=4)
    validate(notebook)
    start = time.perf_counter()
    print(f"Executing {path.relative_to(SOURCE)}", flush=True)
    with TemporaryDirectory(prefix="jeanspy-notebook-") as temporary:
        directory = Path(temporary)
        kernel_dir = directory / "kernels" / "jeanspy-docs"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "JeansPy documentation", "language": "python",
        }))
        manager = KernelManager(kernel_name="jeanspy-docs", kernel_spec_manager=
                                KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
        environment = {**os.environ, "JEANSPY_JAX_PLATFORM": "cpu",
                       "JEANSPY_JAX_ENABLE_X64": "true", "OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "MPLCONFIGDIR": str(directory / "mpl"),
                       "IPYTHONDIR": str(directory / "ipython"),
                       "JUPYTER_PLATFORM_DIRS": "1"}
        # The inline backend retains figures as real notebook outputs.
        environment.pop("MPLBACKEND", None)
        client = NotebookClient(notebook, km=manager, timeout=900, startup_timeout=60,
                                allow_errors=False, record_timing=False,
                                resources={"metadata": {"path": temporary}})
        try:
            client.execute(env=environment, cwd=temporary)
        finally:
            # A caller-supplied KernelManager is not owned by nbclient.
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            if client.kc is not None:
                client.kc.stop_channels()
    notebook.metadata["jeanspy"]["execution"] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": code_hash(notebook),
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "src/jeanspy").rglob("*.py"))},
        "packages": {name: importlib.metadata.version(name) for name in
                     ("jeanspy", "numpy", "scipy", "jax", "numpyro", "arviz", "matplotlib", "corner")},
        "platform": "cpu", "jax_enable_x64": True,
    }
    validate(notebook, saved=True)
    print(f"Executed {path.relative_to(SOURCE)} in {time.perf_counter() - start:.1f}s", flush=True)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-mcmc", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check saved outputs without executing")
    parser.add_argument("--notebook", choices=NOTEBOOKS, action="append")
    args = parser.parse_args()
    for name in args.notebook or NOTEBOOKS:
        path = SOURCE / f"{name}.ipynb"
        notebook = nbformat.read(path, as_version=4)
        validate(notebook, saved=args.check)
        if args.check:
            print(f"Checked {name}")
            continue
        if notebook.metadata["jeanspy"]["mcmc"] and not args.include_mcmc:
            print(f"Skipped MCMC: {name} (requires --include-mcmc)", flush=True)
            continue
        notebook = execute(path)
        if args.write:
            nbformat.write(notebook, path)


if __name__ == "__main__":
    main()
