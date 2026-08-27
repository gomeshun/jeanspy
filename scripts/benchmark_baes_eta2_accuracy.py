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


KERNEL_CONFIGS = [
    (64, 32),
    (128, 32),
    (128, 64),
    (256, 64),
    (256, 128),
    (512, 128),
]
ABEL_CONFIGS = [128, 256, 384, 512, 768, 1024, 1536, 2048]


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


def _bench(jax, func: Callable[[], Any], *, repeats: int, warmups: int) -> float:
    for _ in range(warmups):
        _sync(jax, func())
    return float(statistics.median(_time_once(jax, func) for _ in range(repeats)))


def _max_rel(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / scale))


def _rms_rel(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.abs(b), 1e-12)
    rel = (a - b) / scale
    return float(np.sqrt(np.mean(rel * rel)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure runtime-vs-accuracy convergence for generic Baes, eta=2 "
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

    dtype = jnp.float64
    n_radii = 16 if args.quick else 32
    repeats = args.repeats if args.repeats is not None else (3 if args.quick else 5)
    warmups = 1
    u_max = 5000.0

    kernel_configs = KERNEL_CONFIGS[:4] if args.quick else KERNEL_CONFIGS
    abel_configs = ABEL_CONFIGS[:5] if args.quick else ABEL_CONFIGS

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

    def eta2_kernel_call(n_u: int, n_kernel: int):
        return lambda: eta2_dsph.sigmalos2(
            R,
            params=params_eta2,
            backend="kernel",
            n_u=n_u,
            n_kernel=n_kernel,
            u_max=u_max,
            dm_mass_method="analytic",
            jit=True,
        )

    def generic_kernel_call(n_u: int, n_kernel: int):
        return lambda: generic_dsph.sigmalos2(
            R,
            params=params_generic,
            backend="kernel",
            n_u=n_u,
            n_kernel=n_kernel,
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

    # Independent high-resolution references.  The eta=2 kernel and Abel
    # formulations should converge to the same physical result; reporting their
    # residual difference makes reference bias visible instead of hiding it.
    ref_eta2_fn = eta2_kernel_call(2048 if not args.quick else 1024, 256)
    ref_abel_fn = abel_call(8192 if not args.quick else 4096)
    ref_eta2 = _sync(jax, ref_eta2_fn())
    ref_abel = _sync(jax, ref_abel_fn())
    reference_crosscheck = {
        "eta2_vs_abel_max_rel": _max_rel(ref_eta2, ref_abel),
        "eta2_vs_abel_rms_rel": _rms_rel(ref_eta2, ref_abel),
    }

    rows: list[dict[str, Any]] = []

    for method in ("generic", "eta2"):
        for n_u, n_kernel in kernel_configs:
            fn = (
                generic_kernel_call(n_u, n_kernel)
                if method == "generic"
                else eta2_kernel_call(n_u, n_kernel)
            )
            first_s = _time_once(jax, fn)
            value = _sync(jax, fn())
            hot_s = _bench(jax, fn, repeats=repeats, warmups=warmups)
            rows.append(
                {
                    "method": method,
                    "n_u": n_u,
                    "n_kernel": n_kernel,
                    "n_r": None,
                    "first_s": first_s,
                    "hot_s": hot_s,
                    "max_rel_vs_eta2_ref": _max_rel(value, ref_eta2),
                    "rms_rel_vs_eta2_ref": _rms_rel(value, ref_eta2),
                    "max_rel_vs_abel_ref": _max_rel(value, ref_abel),
                }
            )

    for n_r in abel_configs:
        fn = abel_call(n_r)
        first_s = _time_once(jax, fn)
        value = _sync(jax, fn())
        hot_s = _bench(jax, fn, repeats=repeats, warmups=warmups)
        rows.append(
            {
                "method": "abel",
                "n_u": None,
                "n_kernel": None,
                "n_r": n_r,
                "first_s": first_s,
                "hot_s": hot_s,
                "max_rel_vs_eta2_ref": _max_rel(value, ref_eta2),
                "rms_rel_vs_eta2_ref": _rms_rel(value, ref_eta2),
                "max_rel_vs_abel_ref": _max_rel(value, ref_abel),
            }
        )

    tolerances = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    best_at_tolerance: dict[str, dict[str, Any]] = {}
    for tol in tolerances:
        key = f"{tol:.0e}"
        best_at_tolerance[key] = {}
        for method in ("generic", "eta2", "abel"):
            candidates = [
                row
                for row in rows
                if row["method"] == method and row["max_rel_vs_eta2_ref"] <= tol
            ]
            if candidates:
                best = min(candidates, key=lambda row: row["hot_s"])
                best_at_tolerance[key][method] = best
            else:
                best_at_tolerance[key][method] = None

    print("Matched-accuracy Baes eta=2 benchmark (CPU float64)")
    print(
        "reference cross-check: eta2(high-res) vs Abel(high-res): "
        f"max={reference_crosscheck['eta2_vs_abel_max_rel']:.3e}, "
        f"rms={reference_crosscheck['eta2_vs_abel_rms_rel']:.3e}"
    )
    print()
    print("method   n_u   n_kernel   n_r    hot_ms   max_rel(ref eta2)   rms_rel(ref eta2)")
    for row in rows:
        print(
            f"{row['method']:<8} "
            f"{str(row['n_u']):>5} "
            f"{str(row['n_kernel']):>10} "
            f"{str(row['n_r']):>6} "
            f"{1e3 * row['hot_s']:>9.3f} "
            f"{row['max_rel_vs_eta2_ref']:>18.3e} "
            f"{row['rms_rel_vs_eta2_ref']:>18.3e}"
        )

    print("\nFastest configuration meeting max-relative-error tolerance")
    print("tol       generic_ms   eta2_ms   Abel_ms   eta2/generic   eta2/Abel")
    for tol in tolerances:
        key = f"{tol:.0e}"
        entry = best_at_tolerance[key]
        generic = entry["generic"]
        eta2 = entry["eta2"]
        abel = entry["abel"]
        generic_ms = None if generic is None else 1e3 * generic["hot_s"]
        eta2_ms = None if eta2 is None else 1e3 * eta2["hot_s"]
        abel_ms = None if abel is None else 1e3 * abel["hot_s"]
        eta2_over_generic = (
            None if generic_ms is None or eta2_ms is None else eta2_ms / generic_ms
        )
        eta2_over_abel = None if abel_ms is None or eta2_ms is None else eta2_ms / abel_ms
        print(
            f"{key:<9} "
            f"{('---' if generic_ms is None else f'{generic_ms:.3f}'):>10} "
            f"{('---' if eta2_ms is None else f'{eta2_ms:.3f}'):>9} "
            f"{('---' if abel_ms is None else f'{abel_ms:.3f}'):>9} "
            f"{('---' if eta2_over_generic is None else f'{eta2_over_generic:.3f}x'):>14} "
            f"{('---' if eta2_over_abel is None else f'{eta2_over_abel:.3f}x'):>10}"
        )

    payload = {
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "n_radii": n_radii,
            "u_max": u_max,
            "repeats": repeats,
        },
        "reference_crosscheck": reference_crosscheck,
        "rows": rows,
        "best_at_tolerance": best_at_tolerance,
    }
    print("\nJSON")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
