from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

os.environ.setdefault("JEANSPY_JAX_PLATFORM", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


@dataclass(frozen=True)
class Case:
    name: str
    beta_0: float
    beta_inf: float
    r_a_over_re: float
    rs_over_re: float


CASES = [
    Case("fiducial", -0.5, 0.65, 1.36, 5.0),
    Case("inner-transition", -1.0, 0.8, 0.25, 5.0),
    Case("outer-transition", 0.4, 0.8, 5.0, 5.0),
    Case("tangential", -1.5, -0.2, 1.0, 5.0),
    Case("constant-radial", 0.7, 0.7, 1.0, 2.0),
    Case("broad-halo", -0.5, 0.65, 1.36, 12.0),
]

N_VALUES = [32, 64, 128, 256, 512, 1024, 2048]
TOLERANCES = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]


def _sync(jax, value):
    try:
        return jax.block_until_ready(value)
    except Exception:
        return value


def _time_once(jax, func: Callable[[], Any]) -> float:
    start = time.perf_counter()
    value = func()
    _sync(jax, value)
    return time.perf_counter() - start


def _bench(jax, func: Callable[[], Any], repeats: int = 5) -> float:
    _sync(jax, func())
    return float(statistics.median(_time_once(jax, func) for _ in range(repeats)))


def _max_rel(a, b) -> float:
    import numpy as np

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(aa - bb) / np.maximum(np.abs(bb), 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Robustly benchmark outer-variable transforms for the eta=2 Baes kernel. "
            "The comparison aggregates worst-case accuracy over several anisotropy and halo scales "
            "rather than tuning to one fiducial profile."
        )
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    import numpy as np
    import jax
    import jax.numpy as jnp

    import jeanspy.model_numpyro as mn
    from jeanspy.baes_eta2 import BaesEta2AnisotropyModel
    from jeanspy.model_numpyro import DSphModel, NFWModel, PlummerModel

    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("This benchmark requires JAX float64.")

    dtype = jnp.float64
    re_pc = 220.0
    n_radii = 24 if args.quick else 32
    R = jnp.asarray(np.geomspace(0.02 * re_pc, 5.0 * re_pc, n_radii), dtype=dtype)
    u_max = 5000.0
    n_values = N_VALUES[:5] if args.quick else N_VALUES
    cases = CASES[:4] if args.quick else CASES
    n_ref_outer = 4096 if args.quick else 8192
    n_ref_abel = 16384 if args.quick else 32768
    n_kernel = 32

    def make_dsph() -> DSphModel:
        return DSphModel(
            submodels={
                "StellarModel": PlummerModel(),
                "DMModel": NFWModel(),
                "AnisotropyModel": BaesEta2AnisotropyModel(),
            }
        )

    def params_for(case: Case) -> dict[str, Any]:
        raw = {
            "re_pc": re_pc,
            "rs_pc": case.rs_over_re * re_pc,
            "rhos_Msunpc3": 7.5e-3,
            "r_t_pc": 40.0 * re_pc,
            "beta_0": case.beta_0,
            "beta_inf": case.beta_inf,
            "r_a": case.r_a_over_re * re_pc,
            "vmem_kms": 0.0,
        }
        return {key: jnp.asarray(value, dtype=dtype) for key, value in raw.items()}

    def transformed_kernel_fn(
        dsph: DSphModel,
        params: dict[str, Any],
        *,
        n_u: int,
        transform: str,
    ) -> Callable[[], Any]:
        stellar = dsph.submodels["StellarModel"]
        dm = dsph.submodels["DMModel"]
        ani = dsph.submodels["AnisotropyModel"]

        def _eval():
            R2d = R[:, None]
            sigma2 = stellar.density_2d(R2d, re_pc=params["re_pc"])

            if transform == "sqrtlog":
                s_max = jnp.sqrt(jnp.log(jnp.asarray(u_max, dtype=dtype)))
                x = jnp.linspace(jnp.asarray(0.0, dtype=dtype), s_max, n_u)
                t = x * x
                u = jnp.exp(t)
                jac = 2.0 * x
            elif transform == "q":
                q_max = jnp.sqrt(
                    1.0 - 1.0 / jnp.asarray(u_max * u_max, dtype=dtype)
                )
                x = jnp.linspace(jnp.asarray(0.0, dtype=dtype), q_max, n_u)
                one_minus_q2 = jnp.maximum(1.0 - x * x, jnp.finfo(dtype).tiny)
                u = 1.0 / jnp.sqrt(one_minus_q2)
                jac = x / one_minus_q2  # dt/dq for t=log(u)
            else:
                raise ValueError(transform)

            u2d = u[None, :]
            r = R2d * u2d
            nu3 = stellar.density_3d(r, re_pc=params["re_pc"])
            mass = dm.enclosed_mass(r, method="analytic", params=params)
            grav = (mn.GMsun_m3s2 * mass / mn.PARSEC_M) * 1e-6
            K = ani.kernel(u2d, R2d, params=params, n_kernel=n_kernel)
            base_t = 2.0 * K * (nu3 / sigma2) * grav
            integrand = base_t * jac[None, :]

            if n_u > 1:
                h = x[1] - x[0]
                value = mn._simpson_uniform_last_axis(integrand, h)
            else:
                value = integrand[..., 0]
            return jnp.clip(jnp.nan_to_num(value), 0.0, 1e12)

        return jax.jit(_eval)

    def existing_log_fn(dsph: DSphModel, params: dict[str, Any], n_u: int):
        return lambda: dsph.sigmalos2(
            R,
            params=params,
            backend="kernel",
            n_u=n_u,
            n_kernel=n_kernel,
            u_max=u_max,
            dm_mass_method="analytic",
            jit=True,
        )

    def abel_fn(dsph: DSphModel, params: dict[str, Any], n_r: int):
        return lambda: dsph.sigmalos2(
            R,
            params=params,
            backend="abel",
            n_r=n_r,
            u_max=u_max,
            r_min_factor=0.35,
            dm_mass_method="analytic",
            jit=True,
        )

    case_results: list[dict[str, Any]] = []
    for case in cases:
        dsph = make_dsph()
        params = params_for(case)

        ref_outer_fn = transformed_kernel_fn(
            dsph, params, n_u=n_ref_outer, transform="sqrtlog"
        )
        ref_abel_fn = abel_fn(dsph, params, n_ref_abel)
        ref_outer = _sync(jax, ref_outer_fn())
        ref_abel = _sync(jax, ref_abel_fn())
        ref_cross = _max_rel(ref_outer, ref_abel)

        rows = []
        for n_u in n_values:
            for method in ("log", "sqrtlog", "q"):
                if method == "log":
                    fn = existing_log_fn(dsph, params, n_u)
                else:
                    fn = transformed_kernel_fn(dsph, params, n_u=n_u, transform=method)
                value = _sync(jax, fn())
                hot_s = _bench(jax, fn, repeats=args.repeats)
                err_outer = _max_rel(value, ref_outer)
                err_abel = _max_rel(value, ref_abel)
                rows.append(
                    {
                        "method": method,
                        "n_u": n_u,
                        "hot_s": hot_s,
                        "max_rel_vs_outer_ref": err_outer,
                        "max_rel_vs_abel_ref": err_abel,
                        "conservative_max_rel": max(err_outer, err_abel),
                    }
                )

        case_results.append(
            {
                "case": case.__dict__,
                "reference_crosscheck": ref_cross,
                "rows": rows,
            }
        )

    aggregate_rows = []
    for method in ("log", "sqrtlog", "q"):
        for n_u in n_values:
            matches = [
                row
                for result in case_results
                for row in result["rows"]
                if row["method"] == method and row["n_u"] == n_u
            ]
            aggregate_rows.append(
                {
                    "method": method,
                    "n_u": n_u,
                    "worst_case_max_rel": max(row["conservative_max_rel"] for row in matches),
                    "median_hot_s": float(statistics.median(row["hot_s"] for row in matches)),
                    "max_hot_s": max(row["hot_s"] for row in matches),
                }
            )

    best = {}
    for tol in TOLERANCES:
        key = f"{tol:.0e}"
        best[key] = {}
        for method in ("log", "sqrtlog", "q"):
            candidates = [
                row
                for row in aggregate_rows
                if row["method"] == method and row["worst_case_max_rel"] <= tol
            ]
            best[key][method] = (
                min(candidates, key=lambda row: row["median_hot_s"])
                if candidates
                else None
            )

    worst_ref_cross = max(result["reference_crosscheck"] for result in case_results)
    print("Robust eta=2 outer-integration benchmark (CPU float64)")
    print(
        f"cases={len(cases)}, R/Re=[0.02, 5], "
        f"worst reference cross-check={worst_ref_cross:.3e}"
    )
    print("method    n_u   worst_max_rel   median_hot_ms   max_hot_ms")
    for row in aggregate_rows:
        print(
            f"{row['method']:<8} {row['n_u']:>5} "
            f"{row['worst_case_max_rel']:>15.3e} "
            f"{1e3 * row['median_hot_s']:>15.3f} "
            f"{1e3 * row['max_hot_s']:>12.3f}"
        )

    print("\nFastest common configuration meeting worst-case tolerance")
    print("tol       log_ms   sqrtlog_ms   q_ms   sqrtlog/log")
    for tol in TOLERANCES:
        key = f"{tol:.0e}"
        entry = best[key]
        values = {}
        for method in ("log", "sqrtlog", "q"):
            values[method] = (
                None if entry[method] is None else 1e3 * entry[method]["median_hot_s"]
            )
        ratio = (
            None
            if values["log"] is None or values["sqrtlog"] is None
            else values["sqrtlog"] / values["log"]
        )
        fmt = lambda x: "---" if x is None else f"{x:.3f}"
        print(
            f"{key:<9} {fmt(values['log']):>7} {fmt(values['sqrtlog']):>12} "
            f"{fmt(values['q']):>7} {('---' if ratio is None else f'{ratio:.3f}x'):>13}"
        )

    payload = {
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "n_radii": n_radii,
            "u_max": u_max,
            "n_kernel": n_kernel,
            "n_ref_outer": n_ref_outer,
            "n_ref_abel": n_ref_abel,
        },
        "worst_reference_crosscheck": worst_ref_cross,
        "aggregate_rows": aggregate_rows,
        "best_at_tolerance": best,
        "case_results": case_results,
    }
    print("\nJSON")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
