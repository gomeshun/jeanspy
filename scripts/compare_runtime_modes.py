from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast


MODES = {
    "scipy": {
        "JEANSPY_HYP2F1_BACKEND": "scipy",
        "JEANSPY_JAX_PLATFORM": "cpu",
        "JEANSPY_JAX_ENABLE_X64": "true",
    },
    "jax-cpu": {
        "JEANSPY_HYP2F1_BACKEND": "jax",
        "JEANSPY_JAX_PLATFORM": "cpu",
        "JEANSPY_JAX_ENABLE_X64": "false",
    },
    "jax-gpu-x32": {
        "JEANSPY_HYP2F1_BACKEND": "jax",
        "JEANSPY_JAX_PLATFORM": "gpu",
        "JEANSPY_JAX_ENABLE_X64": "false",
    },
    "jax-gpu-x64": {
        "JEANSPY_HYP2F1_BACKEND": "jax",
        "JEANSPY_JAX_PLATFORM": "gpu",
        "JEANSPY_JAX_ENABLE_X64": "true",
    },
}


def _median(values: list[float]) -> float:
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _worker(mode_name: str) -> None:
    if "JEANSPY_JAX_PLATFORM" in os.environ:
        requested_platform = os.environ["JEANSPY_JAX_PLATFORM"].strip().lower()
        os.environ["JAX_PLATFORMS"] = "cuda" if requested_platform == "gpu" else requested_platform
    if "JEANSPY_JAX_ENABLE_X64" in os.environ:
        os.environ["JAX_ENABLE_X64"] = os.environ["JEANSPY_JAX_ENABLE_X64"]

    import numpy as np
    import jax
    import jax.numpy as jnp

    import jeanspy.model as legacy_mod
    import jeanspy.model_numpyro as new_mod

    def sync(value):
        try:
            return jax.block_until_ready(value)
        except Exception:
            return value

    def time_once(func):
        start = time.perf_counter()
        out = func()
        sync(out)
        return time.perf_counter() - start

    def bench(func, *, repeats: int, warmups: int) -> float:
        for _ in range(warmups):
            sync(func())
        return _median([time_once(func) for _ in range(repeats)])

    def make_legacy_dsph(params: dict[str, float]):
        if {"a", "b", "g"} <= params.keys():
            dm = legacy_mod.ZhaoModel(
                rs_pc=params["rs_pc"],
                rhos_Msunpc3=params["rhos_Msunpc3"],
                a=params["a"],
                b=params["b"],
                g=params["g"],
                r_t_pc=params["r_t_pc"],
            )
        else:
            dm = legacy_mod.NFWModel(
                rs_pc=params["rs_pc"],
                rhos_Msunpc3=params["rhos_Msunpc3"],
                r_t_pc=params["r_t_pc"],
            )

        return legacy_mod.DSphModel(
            vmem_kms=params["vmem_kms"],
            submodels={
                "StellarModel": legacy_mod.PlummerModel(re_pc=params["re_pc"]),
                "DMModel": dm,
                "AnisotropyModel": legacy_mod.ConstantAnisotropyModel(beta_ani=params["beta_ani"]),
            },
        )

    def make_new_dsph():
        return new_mod.DSphModel(
            submodels={
                "StellarModel": new_mod.PlummerModel(),
                "DMModel": new_mod.NFWModel(),
                "AnisotropyModel": new_mod.ConstantAnisotropyModel(),
            }
        )

    def make_new_zhao_dsph():
        return new_mod.DSphModel(
            submodels={
                "StellarModel": new_mod.PlummerModel(),
                "DMModel": new_mod.ZhaoModel(),
                "AnisotropyModel": new_mod.ConstantAnisotropyModel(),
            }
        )

    def kernel_accuracy_wide_beta() -> float:
        betas = np.array([-10.0, -5.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9, 1.0], dtype=np.float64)
        u_np = np.geomspace(1.0 + 1e-4, 5e2, 220).astype(np.float64)
        u_legacy = u_np[None, :]
        u_jax = jnp.asarray(u_np, dtype=jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32)[None, :]
        R_legacy = np.asarray([100.0], dtype=np.float64)[:, None]
        R_jax = jnp.asarray([100.0], dtype=u_jax.dtype)[:, None]

        new_const = new_mod.ConstantAnisotropyModel()
        max_rel = 0.0
        for beta in betas:
            legacy_const = legacy_mod.ConstantAnisotropyModel(beta_ani=float(beta))
            ref = np.asarray(legacy_const.kernel(u_legacy, R_legacy), dtype=np.float64).reshape(-1)
            got = np.asarray(
                new_const.kernel(u_jax, R_jax, params={"beta_ani": u_jax.dtype.type(beta)}),
                dtype=np.float64,
            ).reshape(-1)
            rel = np.max(np.abs(got - ref) / (np.abs(ref) + 1e-300))
            max_rel = max(max_rel, float(rel))
        return max_rel

    def sigmalos2_accuracy(
        params: dict[str, float],
        *,
        n_legacy: int,
        n_kernel: int,
        n_u: int | None = None,
        u_max: float | None = None,
    ) -> float:
        legacy_dsph = make_legacy_dsph(params)
        if {"a", "b", "g"} <= params.keys():
            new_dsph = make_new_zhao_dsph()
        else:
            new_dsph = make_new_dsph()

        R_pc = np.geomspace(max(0.1, params["re_pc"] * 0.05), params["re_pc"] * 5.0, 10).astype(np.float64)
        ref = np.asarray(legacy_dsph.sigmalos2_dequad(R_pc, n=n_legacy, n_kernel=n_kernel), dtype=np.float64)
        dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
        params_jax = {key: jnp.asarray(value, dtype=dtype) for key, value in params.items()}
        r_pc_jax = jnp.asarray(R_pc, dtype=dtype)
        if n_u is None and u_max is None:
            got = np.asarray(
                new_dsph.sigmalos2(r_pc_jax, params=params_jax, backend="kernel", jit=True),
                dtype=np.float64,
            )
        elif n_u is None:
            got = np.asarray(
                new_dsph.sigmalos2(r_pc_jax, params=params_jax, backend="kernel", jit=True, u_max=u_max),
                dtype=np.float64,
            )
        elif u_max is None:
            got = np.asarray(
                new_dsph.sigmalos2(r_pc_jax, params=params_jax, backend="kernel", jit=True, n_u=n_u),
                dtype=np.float64,
            )
        else:
            got = np.asarray(
                new_dsph.sigmalos2(r_pc_jax, params=params_jax, backend="kernel", jit=True, n_u=n_u, u_max=u_max),
                dtype=np.float64,
            )
        return float(np.max(np.abs(got - ref) / (np.abs(ref) + 1e-300)))

    params_typical = {
        "re_pc": 200.0,
        "rs_pc": 1200.0,
        "rhos_Msunpc3": 1e-2,
        "r_t_pc": 8000.0,
        "beta_ani": 0.2,
        "vmem_kms": 0.0,
    }
    params_boundary = {
        "re_pc": 50.0,
        "rs_pc": 300.0,
        "rhos_Msunpc3": 0.2,
        "a": 0.7,
        "b": 5.5,
        "g": 1.2,
        "r_t_pc": 2e4,
        "beta_ani": 0.95,
        "vmem_kms": 0.0,
    }

    legacy_dsph = make_legacy_dsph(params_typical)
    new_dsph = make_new_dsph()
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    params_jax = {key: jnp.asarray(value, dtype=dtype) for key, value in params_typical.items()}

    u_kernel_np = np.geomspace(1.0 + 1e-4, 1e4, 4096).astype(np.float64)
    u_kernel_legacy = u_kernel_np[None, :]
    u_kernel_jax = jnp.asarray(u_kernel_np, dtype=dtype)[None, :]
    R_kernel_legacy = np.asarray([100.0], dtype=np.float64)[:, None]
    R_kernel_jax = jnp.asarray([100.0], dtype=dtype)[:, None]
    R_sig_np = np.geomspace(10.0, 1000.0, 24).astype(np.float64)
    R_sig_jax = jnp.asarray(R_sig_np, dtype=dtype)

    legacy_kernel_fn = lambda: legacy_dsph["AnisotropyModel"].kernel(u_kernel_legacy, R_kernel_legacy)
    legacy_sig_fn = lambda: legacy_dsph.sigmalos2_dequad(R_sig_np, n=1024, n_kernel=128)
    new_kernel_model = cast(Any, new_dsph.submodels["AnisotropyModel"])
    new_kernel_fn = lambda: new_kernel_model.kernel(u_kernel_jax, R_kernel_jax, params=params_jax)
    new_sig_eager = lambda: new_dsph.sigmalos2(R_sig_jax, params=params_jax, backend="kernel", jit=False)
    new_sig_jit = lambda: new_dsph.sigmalos2(R_sig_jax, params=params_jax, backend="kernel", jit=True)

    result = {
        "mode": mode_name,
        "requested_env": MODES[mode_name],
        "runtime": new_mod.get_runtime_config(),
        "accuracy": {
            "kernel_wide_beta_max_rel": kernel_accuracy_wide_beta(),
            "sigmalos2_typical_max_rel": sigmalos2_accuracy(
                params_typical,
                n_legacy=2048,
                n_kernel=256,
            ),
            "sigmalos2_boundary_max_rel": sigmalos2_accuracy(
                params_boundary,
                n_legacy=2048,
                n_kernel=256,
            ),
        },
        "speed_s": {
            "model_py_kernel_hot_median": bench(legacy_kernel_fn, repeats=5, warmups=1),
            "model_numpyro_kernel_first": time_once(new_kernel_fn),
            "model_numpyro_kernel_hot_median": bench(new_kernel_fn, repeats=5, warmups=1),
            "model_py_sigmalos2_hot_median": bench(legacy_sig_fn, repeats=3, warmups=0),
            "model_numpyro_sigmalos2_eager_first": time_once(new_sig_eager),
            "model_numpyro_sigmalos2_eager_hot_median": bench(new_sig_eager, repeats=3, warmups=1),
            "model_numpyro_sigmalos2_jit_first": time_once(new_sig_jit),
            "model_numpyro_sigmalos2_jit_hot_median": bench(new_sig_jit, repeats=5, warmups=1),
        },
    }

    speed = result["speed_s"]
    result["speedup_vs_model_py"] = {
        "kernel_hot": speed["model_py_kernel_hot_median"] / speed["model_numpyro_kernel_hot_median"],
        "sigmalos2_eager_hot": speed["model_py_sigmalos2_hot_median"] / speed["model_numpyro_sigmalos2_eager_hot_median"],
        "sigmalos2_jit_hot": speed["model_py_sigmalos2_hot_median"] / speed["model_numpyro_sigmalos2_jit_hot_median"],
    }

    print(json.dumps(result, sort_keys=True))


def _launch_worker(mode_name: str) -> dict[str, object]:
    env = os.environ.copy()
    env.update(MODES[mode_name])
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", mode_name],
        cwd=repo_root,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return {
            "mode": mode_name,
            "error": proc.stderr.strip() or proc.stdout.strip() or f"worker failed with exit code {proc.returncode}",
        }
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def _print_summary(results: list[dict[str, object]]) -> None:
    print("Accuracy (max relative error vs legacy/model.py reference)")
    print("mode                 kernel        sig_typical    sig_boundary")
    for result in results:
        if "error" in result:
            print(f"{result['mode']:<20} ERROR         ERROR          ERROR")
            continue
        accuracy = cast(dict[str, Any], result["accuracy"])
        print(
            f"{result['mode']:<20} "
            f"{accuracy['kernel_wide_beta_max_rel']:<13.3e} "
            f"{accuracy['sigmalos2_typical_max_rel']:<14.3e} "
            f"{accuracy['sigmalos2_boundary_max_rel']:<14.3e}"
        )

    print()
    print("Speed (seconds, median hot-call timing unless marked first)")
    print("mode                 kernel_hot    sig_eager_hot  sig_jit_hot   speedup_jit_vs_model.py")
    for result in results:
        if "error" in result:
            print(f"{result['mode']:<20} ERROR         ERROR          ERROR         ERROR")
            continue
        speed = cast(dict[str, Any], result["speed_s"])
        speedup = cast(dict[str, Any], result["speedup_vs_model_py"])
        print(
            f"{result['mode']:<20} "
            f"{speed['model_numpyro_kernel_hot_median']:<13.3e} "
            f"{speed['model_numpyro_sigmalos2_eager_hot_median']:<14.3e} "
            f"{speed['model_numpyro_sigmalos2_jit_hot_median']:<13.3e} "
            f"{speedup['sigmalos2_jit_hot']:<.3f}"
        )

    print()
    print("Full JSON")
    print(json.dumps(results, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare JeansPy runtime modes.")
    parser.add_argument("--worker", choices=sorted(MODES), help="internal worker mode")
    args = parser.parse_args()

    if args.worker:
        _worker(args.worker)
        return

    results = [_launch_worker(mode_name) for mode_name in MODES]
    _print_summary(results)


if __name__ == "__main__":
    main()