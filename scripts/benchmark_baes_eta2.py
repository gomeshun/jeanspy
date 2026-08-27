from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MODES = {
    "cpu-x32": {"JAX_ENABLE_X64": "false"},
    "cpu-x64": {"JAX_ENABLE_X64": "true"},
}


def _sync(jax, value):
    try:
        return jax.block_until_ready(value)
    except Exception:
        return value


def _time_once(jax, func) -> float:
    start = time.perf_counter()
    out = func()
    _sync(jax, out)
    return time.perf_counter() - start


def _bench(jax, func, *, repeats: int, warmups: int) -> float:
    for _ in range(warmups):
        _sync(jax, func())
    return float(statistics.median(_time_once(jax, func) for _ in range(repeats)))


def _max_rel(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def _worker(mode_name: str, *, quick: bool) -> None:
    import numpy as np
    import jax
    import jax.numpy as jnp

    from jeanspy.baes_eta2 import BaesEta2AnisotropyModel
    from jeanspy.model_numpyro import (
        BaesAnisotropyModel,
        DSphModel,
        NFWModel,
        PlummerModel,
        get_runtime_config,
    )

    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

    if quick:
        n_radii = 16
        n_u = 128
        n_kernel = 96
        n_r_abel = 512
        repeats = 3
        warmups = 1
        n_kernel_u = 320
    else:
        n_radii = 32
        n_u = 256
        n_kernel = 128
        n_r_abel = 768
        repeats = 7
        warmups = 2
        n_kernel_u = 768

    params_common = {
        "re_pc": 220.0,
        "rs_pc": 1100.0,
        "rhos_Msunpc3": 7.5e-3,
        "r_t_pc": 9000.0,
        "beta_0": -0.5,
        "beta_inf": 0.65,
        "r_a": 300.0,
        "vmem_kms": 0.0,
    }
    params_eta2 = {key: jnp.asarray(value, dtype=dtype) for key, value in params_common.items()}
    params_generic = dict(params_eta2)
    params_generic["eta"] = jnp.asarray(2.0, dtype=dtype)

    generic_ani = BaesAnisotropyModel()
    eta2_ani = BaesEta2AnisotropyModel()

    u = jnp.asarray(np.geomspace(1.0 + 1e-4, 100.0, n_kernel_u), dtype=dtype)[None, :]
    R_kernel = jnp.asarray(np.geomspace(10.0, 1200.0, n_radii), dtype=dtype)[:, None]

    generic_kernel = lambda: generic_ani.kernel(
        u, R_kernel, params=params_generic, n_kernel=n_kernel
    )
    eta2_kernel = lambda: eta2_ani.kernel(
        u, R_kernel, params=params_eta2, n_kernel=n_kernel
    )

    # Compile both paths before the accuracy comparison so conversion to NumPy
    # does not contaminate the timing section.
    k_generic = _sync(jax, generic_kernel())
    k_eta2 = _sync(jax, eta2_kernel())
    kernel_max_rel = _max_rel(k_eta2, k_generic)

    def make_dsph(anisotropy):
        return DSphModel(
            submodels={
                "StellarModel": PlummerModel(),
                "DMModel": NFWModel(),
                "AnisotropyModel": anisotropy,
            }
        )

    generic_dsph = make_dsph(BaesAnisotropyModel())
    eta2_dsph = make_dsph(BaesEta2AnisotropyModel())
    R = jnp.asarray(np.geomspace(5.0, 900.0, n_radii), dtype=dtype)

    sig_generic = lambda: generic_dsph.sigmalos2(
        R,
        params=params_generic,
        backend="kernel",
        n_u=n_u,
        n_kernel=n_kernel,
        u_max=1600.0,
        dm_mass_method="analytic",
        jit=True,
    )
    sig_eta2 = lambda: eta2_dsph.sigmalos2(
        R,
        params=params_eta2,
        backend="kernel",
        n_u=n_u,
        n_kernel=n_kernel,
        u_max=1600.0,
        dm_mass_method="analytic",
        jit=True,
    )
    sig_abel = lambda: eta2_dsph.sigmalos2(
        R,
        params=params_eta2,
        backend="abel",
        n_r=n_r_abel,
        u_max=1600.0,
        r_min_factor=0.35,
        dm_mass_method="analytic",
        jit=True,
    )

    # First-call timings include tracing/XLA compilation.
    first_generic = _time_once(jax, sig_generic)
    first_eta2 = _time_once(jax, sig_eta2)
    first_abel = _time_once(jax, sig_abel)

    s_generic = _sync(jax, sig_generic())
    s_eta2 = _sync(jax, sig_eta2())
    s_abel = _sync(jax, sig_abel())

    result: dict[str, Any] = {
        "mode": mode_name,
        "quick": quick,
        "runtime": get_runtime_config(),
        "shape": {
            "kernel_R": n_radii,
            "kernel_u": n_kernel_u,
            "sigmalos_R": n_radii,
            "n_u": n_u,
            "n_kernel": n_kernel,
            "n_r_abel": n_r_abel,
        },
        "accuracy": {
            "kernel_eta2_vs_generic_max_rel": kernel_max_rel,
            "sigmalos_eta2_vs_generic_max_rel": _max_rel(s_eta2, s_generic),
            "sigmalos_eta2_vs_abel_max_rel": _max_rel(s_eta2, s_abel),
        },
        "timing_s": {
            "kernel_generic_first": _time_once(jax, generic_kernel),
            "kernel_eta2_first": _time_once(jax, eta2_kernel),
            "kernel_generic_hot_median": _bench(
                jax, generic_kernel, repeats=repeats, warmups=warmups
            ),
            "kernel_eta2_hot_median": _bench(
                jax, eta2_kernel, repeats=repeats, warmups=warmups
            ),
            "sigmalos_generic_first": first_generic,
            "sigmalos_eta2_first": first_eta2,
            "sigmalos_abel_first": first_abel,
            "sigmalos_generic_hot_median": _bench(
                jax, sig_generic, repeats=repeats, warmups=warmups
            ),
            "sigmalos_eta2_hot_median": _bench(
                jax, sig_eta2, repeats=repeats, warmups=warmups
            ),
            "sigmalos_abel_hot_median": _bench(
                jax, sig_abel, repeats=repeats, warmups=warmups
            ),
        },
    }

    t = result["timing_s"]
    result["speedup"] = {
        "kernel_eta2_vs_generic_hot": t["kernel_generic_hot_median"]
        / t["kernel_eta2_hot_median"],
        "sigmalos_eta2_vs_generic_hot": t["sigmalos_generic_hot_median"]
        / t["sigmalos_eta2_hot_median"],
        "sigmalos_eta2_vs_abel_hot": t["sigmalos_abel_hot_median"]
        / t["sigmalos_eta2_hot_median"],
    }

    print(json.dumps(result, sort_keys=True))


def _launch_worker(mode_name: str, *, quick: bool) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(MODES[mode_name])
    env["JEANSPY_JAX_PLATFORM"] = "cpu"
    env["JAX_PLATFORMS"] = "cpu"
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        mode_name,
    ]
    if quick:
        command.append("--quick")

    proc = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return {
            "mode": mode_name,
            "error": proc.stderr.strip() or proc.stdout.strip(),
        }
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("Baes eta=2 benchmark")
    print("mode      kernel hot [ms]  kernel speedup  sig generic [ms]  sig eta2 [ms]  sig Abel [ms]  eta2/generic")
    for result in results:
        if "error" in result:
            print(f"{result['mode']:<9} ERROR: {result['error']}")
            continue
        timing = result["timing_s"]
        speedup = result["speedup"]
        print(
            f"{result['mode']:<9} "
            f"{1e3 * timing['kernel_eta2_hot_median']:>14.3f}  "
            f"{speedup['kernel_eta2_vs_generic_hot']:>14.3f}x  "
            f"{1e3 * timing['sigmalos_generic_hot_median']:>16.3f}  "
            f"{1e3 * timing['sigmalos_eta2_hot_median']:>12.3f}  "
            f"{1e3 * timing['sigmalos_abel_hot_median']:>12.3f}  "
            f"{speedup['sigmalos_eta2_vs_generic_hot']:>12.3f}x"
        )

    print()
    print("Accuracy")
    print("mode      kernel eta2/generic   sig eta2/generic   sig eta2/Abel")
    for result in results:
        if "error" in result:
            continue
        accuracy = result["accuracy"]
        print(
            f"{result['mode']:<9} "
            f"{accuracy['kernel_eta2_vs_generic_max_rel']:>19.3e}  "
            f"{accuracy['sigmalos_eta2_vs_generic_max_rel']:>17.3e}  "
            f"{accuracy['sigmalos_eta2_vs_abel_max_rel']:>14.3e}"
        )

    print("\nJSON")
    print(json.dumps(results, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the eta=2 Baes--van Hese analytic reduction against "
            "the generic numerical BAES kernel and the kernel-free Abel solver."
        )
    )
    parser.add_argument(
        "--modes",
        default="cpu-x32,cpu-x64",
        help="Comma-separated subset of: cpu-x32,cpu-x64",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller grids/repeat counts for CI or smoke benchmarking.",
    )
    parser.add_argument("--worker", choices=tuple(MODES), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(args.worker, quick=bool(args.quick))
        return

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        parser.error(f"unknown modes: {', '.join(unknown)}")

    results = [_launch_worker(mode, quick=bool(args.quick)) for mode in modes]
    _print_summary(results)
    if any("error" in result for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
