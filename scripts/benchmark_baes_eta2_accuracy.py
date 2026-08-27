from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any, Callable

os.environ.setdefault("JEANSPY_JAX_PLATFORM", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

KERNEL_N_U = [64, 128, 256, 512, 1024, 2048, 4096]
ABEL_N_R = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
TOLERANCES = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]


def _sync(jax, value):
    try:
        return jax.block_until_ready(value)
    except Exception:
        return value


def _time_once(jax, func: Callable[[], Any]) -> float:
    start = time.perf_counter()
    out = func()
    _sync(jax, out)
    return time.perf_counter() - start


def _bench(jax, func: Callable[[], Any], *, repeats: int) -> float:
    _sync(jax, func())
    return float(statistics.median(_time_once(jax, func) for _ in range(repeats)))


def _max_rel(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def _rms_rel(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rel = (a - b) / np.maximum(np.abs(b), 1e-12)
    return float(np.sqrt(np.mean(rel * rel)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Matched-accuracy convergence benchmark for generic Baes, eta=2 "
            "Baes, and kernel-free Abel sigma_los^2 backends."
        )
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=None)
    args = parser.parse_args()

    import numpy as np
    import jax
    import jax.numpy as jnp

    from jeanspy.baes_eta2 import BaesEta2AnisotropyModel
    from jeanspy.model_numpyro import BaesAnisotropyModel, DSphModel, NFWModel, PlummerModel

    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("This convergence benchmark requires JAX float64.")

    n_radii = 16 if args.quick else 32
    repeats = args.repeats if args.repeats is not None else (3 if args.quick else 5)
    u_max = 5000.0
    n_kernel = 32
    kernel_n_u = KERNEL_N_U[:5] if args.quick else KERNEL_N_U
    abel_n_r = ABEL_N_R[:6] if args.quick else ABEL_N_R

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
    params_eta2 = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in params_common.items()}
    params_generic = dict(params_eta2)
    params_generic["eta"] = jnp.asarray(2.0, dtype=jnp.float64)

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
    R = jnp.asarray(np.geomspace(5.0, 900.0, n_radii), dtype=jnp.float64)

    def kernel_call(method: str, n_u: int, n_kernel_local: int = n_kernel):
        if method == "generic":
            model, params = generic_dsph, params_generic
        else:
            model, params = eta2_dsph, params_eta2
        return lambda: model.sigmalos2(
            R,
            params=params,
            backend="kernel",
            n_u=n_u,
            n_kernel=n_kernel_local,
            u_max=u_max,
            dm_mass_method="analytic",
            jit=True,
        )

    def abel_call(n_r: int):
        return lambda: eta2_dsph.sigmalos2(
            R,
            params=params_eta2,
            backend="abel",
            n_r=n_r,
            u_max=u_max,
            r_min_factor=0.35,
            dm_mass_method="analytic",
            jit=True,
        )

    # Two independently converged references.  Matched-accuracy decisions use
    # the worse error against these two references so that neither formulation
    # is privileged when their residual numerical difference is non-zero.
    ref_eta2_fn = kernel_call("eta2", 4096 if args.quick else 8192, 64)
    ref_abel_fn = abel_call(16384 if args.quick else 32768)
    ref_eta2 = _sync(jax, ref_eta2_fn())
    ref_abel = _sync(jax, ref_abel_fn())
    reference_crosscheck = {
        "eta2_vs_abel_max_rel": _max_rel(ref_eta2, ref_abel),
        "eta2_vs_abel_rms_rel": _rms_rel(ref_eta2, ref_abel),
    }

    rows: list[dict[str, Any]] = []

    def record(method: str, fn: Callable[[], Any], *, n_u=None, n_r=None):
        first_s = _time_once(jax, fn)
        value = _sync(jax, fn())
        hot_s = _bench(jax, fn, repeats=repeats)
        err_eta2 = _max_rel(value, ref_eta2)
        err_abel = _max_rel(value, ref_abel)
        rows.append(
            {
                "method": method,
                "n_u": n_u,
                "n_kernel": n_kernel if n_u is not None else None,
                "n_r": n_r,
                "first_s": first_s,
                "hot_s": hot_s,
                "max_rel_vs_eta2_ref": err_eta2,
                "max_rel_vs_abel_ref": err_abel,
                "conservative_max_rel": max(err_eta2, err_abel),
                "rms_rel_vs_eta2_ref": _rms_rel(value, ref_eta2),
            }
        )

    for method in ("generic", "eta2"):
        for n_u in kernel_n_u:
            record(method, kernel_call(method, n_u), n_u=n_u)

    for n_r in abel_n_r:
        record("abel", abel_call(n_r), n_r=n_r)

    # Verify explicitly that the inner eta=2 kernel quadrature is already
    # converged at n_kernel=32 for a representative outer grid.
    nk_check = {}
    for nk in (32, 64, 128):
        value = _sync(jax, kernel_call("eta2", 512, nk)())
        nk_check[str(nk)] = _max_rel(value, ref_eta2)

    best_at_tolerance: dict[str, dict[str, Any]] = {}
    for tol in TOLERANCES:
        key = f"{tol:.0e}"
        best_at_tolerance[key] = {}
        for method in ("generic", "eta2", "abel"):
            candidates = [
                row
                for row in rows
                if row["method"] == method and row["conservative_max_rel"] <= tol
            ]
            best_at_tolerance[key][method] = (
                min(candidates, key=lambda row: row["hot_s"]) if candidates else None
            )

    print("Matched-accuracy Baes eta=2 benchmark (CPU float64)")
    print(
        "reference cross-check: eta2(high-res) vs Abel(high-res): "
        f"max={reference_crosscheck['eta2_vs_abel_max_rel']:.3e}, "
        f"rms={reference_crosscheck['eta2_vs_abel_rms_rel']:.3e}"
    )
    print("eta2 n_kernel convergence at n_u=512:", nk_check)
    print()
    print("method   n_u    n_r     hot_ms   conservative_max_rel")
    for row in rows:
        print(
            f"{row['method']:<8} {str(row['n_u']):>5} {str(row['n_r']):>7} "
            f"{1e3 * row['hot_s']:>10.3f} {row['conservative_max_rel']:>22.3e}"
        )

    print("\nFastest configuration meeting conservative max-relative-error tolerance")
    print("tol       generic_ms   eta2_ms   Abel_ms   eta2/generic   eta2/Abel")
    for tol in TOLERANCES:
        key = f"{tol:.0e}"
        entry = best_at_tolerance[key]
        times = {
            method: None if entry[method] is None else 1e3 * entry[method]["hot_s"]
            for method in ("generic", "eta2", "abel")
        }
        eg = None if times["generic"] is None or times["eta2"] is None else times["eta2"] / times["generic"]
        ea = None if times["abel"] is None or times["eta2"] is None else times["eta2"] / times["abel"]
        fmt = lambda x: "---" if x is None else f"{x:.3f}"
        fmtr = lambda x: "---" if x is None else f"{x:.3f}x"
        print(
            f"{key:<9} {fmt(times['generic']):>10} {fmt(times['eta2']):>9} "
            f"{fmt(times['abel']):>9} {fmtr(eg):>14} {fmtr(ea):>10}"
        )

    print("\nJSON")
    print(
        json.dumps(
            {
                "runtime": {
                    "jax_backend": jax.default_backend(),
                    "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
                    "n_radii": n_radii,
                    "u_max": u_max,
                    "repeats": repeats,
                },
                "reference_crosscheck": reference_crosscheck,
                "n_kernel_check": nk_check,
                "rows": rows,
                "best_at_tolerance": best_at_tolerance,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
