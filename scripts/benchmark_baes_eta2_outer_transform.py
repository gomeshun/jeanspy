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
    note: str


# This is deliberately an adversarial numerical set, not a prior or a catalogue
# of physically preferred models.  MCMC walkers can visit awkward regions while
# exploring, so the transform should not be tuned only to a fiducial dSph.
CASES = [
    Case("fiducial", -0.5, 0.65, 1.36, 5.0, "representative transition"),
    Case("inner-transition", -1.0, 0.8, 0.25, 5.0, "transition inside Re"),
    Case("outer-transition", 0.4, 0.8, 5.0, 5.0, "transition outside Re"),
    Case("moderate-tangential", -1.5, -0.2, 1.0, 5.0, "tangential profile"),
    Case("constant-radial", 0.7, 0.7, 1.0, 2.0, "constant radial limit"),
    Case("near-radial-constant", 0.98, 0.98, 1.0, 5.0, "beta -> 1 cancellation"),
    Case("extreme-tangential-constant", -5.0, -5.0, 1.0, 5.0, "large negative beta dynamic range"),
    Case("rapid-radializing", -5.0, 0.98, 0.01, 5.0, "large beta contrast and small ra"),
    Case("rapid-tangentializing", 0.98, -5.0, 0.01, 5.0, "reversed large beta contrast"),
    Case("tiny-ra", -0.5, 0.98, 1e-3, 5.0, "almost outer-anisotropy limit"),
    Case("huge-ra", 0.98, -0.5, 1e3, 5.0, "almost inner-anisotropy limit"),
    Case("compact-halo", -0.5, 0.65, 1.36, 0.05, "rs well inside Re"),
    Case("broad-halo", -0.5, 0.65, 1.36, 12.0, "rs well outside Re"),
    Case("ultra-broad-halo", -0.5, 0.65, 1.36, 100.0, "very slowly varying halo mass"),
]

N_VALUES = [32, 64, 128, 256, 512, 1024, 2048]
TOLERANCES = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
REFERENCE_WARN = 3e-4


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


def _error_summary(a, b) -> dict[str, float | bool]:
    import numpy as np

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    finite = bool(np.isfinite(aa).all() and np.isfinite(bb).all())
    if not finite:
        return {
            "finite": False,
            "max_rel": float("inf"),
            "max_scaled_abs": float("inf"),
            "near_zero_fraction": 1.0,
        }

    scale = max(float(np.max(np.abs(bb))), 1e-12)
    floor = max(scale * 1e-8, 1e-12)
    absdiff = np.abs(aa - bb)
    max_rel = float(np.max(absdiff / np.maximum(np.abs(bb), floor)))
    max_scaled_abs = float(np.max(absdiff) / scale)
    near_zero_fraction = float(np.mean(np.abs(bb) < floor))
    return {
        "finite": True,
        "max_rel": max_rel,
        "max_scaled_abs": max_scaled_abs,
        "near_zero_fraction": near_zero_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Robustly benchmark outer-variable transforms for the eta=2 Baes kernel. "
            "The comparison aggregates worst-case accuracy over representative and "
            "MCMC-adversarial anisotropy/halo scales instead of tuning one profile."
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
    n_radii = 28 if args.quick else 40
    # Wider than the earlier fiducial test.  This probes R far inside/outside
    # tracer and anisotropy scales without going so small that u_max truncation
    # trivially dominates the reference integral.
    R = jnp.asarray(np.geomspace(0.005 * re_pc, 10.0 * re_pc, n_radii), dtype=dtype)
    u_max = 5000.0
    n_values = N_VALUES[:5] if args.quick else N_VALUES
    cases = CASES[:8] if args.quick else CASES
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
                jac = x / one_minus_q2
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
        ref_summary = _error_summary(ref_outer, ref_abel)

        rows = []
        for n_u in n_values:
            for method in ("log", "sqrtlog", "q"):
                if method == "log":
                    fn = existing_log_fn(dsph, params, n_u)
                else:
                    fn = transformed_kernel_fn(dsph, params, n_u=n_u, transform=method)
                value = _sync(jax, fn())
                hot_s = _bench(jax, fn, repeats=args.repeats)
                err_outer = _error_summary(value, ref_outer)
                err_abel = _error_summary(value, ref_abel)
                conservative = max(float(err_outer["max_rel"]), float(err_abel["max_rel"]))
                rows.append(
                    {
                        "method": method,
                        "n_u": n_u,
                        "hot_s": hot_s,
                        "max_rel_vs_outer_ref": err_outer["max_rel"],
                        "max_rel_vs_abel_ref": err_abel["max_rel"],
                        "conservative_max_rel": conservative,
                        "finite": bool(err_outer["finite"] and err_abel["finite"]),
                    }
                )

        case_results.append(
            {
                "case": case.__dict__,
                "reference_crosscheck": ref_summary,
                "reference_warning": (
                    (not bool(ref_summary["finite"]))
                    or float(ref_summary["max_rel"]) > REFERENCE_WARN
                ),
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
                    "all_finite": all(row["finite"] for row in matches),
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
                if row["all_finite"] and row["worst_case_max_rel"] <= tol
            ]
            best[key][method] = (
                min(candidates, key=lambda row: row["median_hot_s"])
                if candidates
                else None
            )

    worst_ref_cross = max(
        float(result["reference_crosscheck"]["max_rel"]) for result in case_results
    )
    warned_cases = [result["case"]["name"] for result in case_results if result["reference_warning"]]
    print("Robust eta=2 outer-integration benchmark (CPU float64)")
    print(
        f"cases={len(cases)}, R/Re=[0.005, 10], "
        f"worst reference cross-check={worst_ref_cross:.3e}"
    )
    print(f"reference-warning cases: {warned_cases or 'none'}")
    print("method    n_u   worst_max_rel   finite   median_hot_ms   max_hot_ms")
    for row in aggregate_rows:
        print(
            f"{row['method']:<8} {row['n_u']:>5} "
            f"{row['worst_case_max_rel']:>15.3e} "
            f"{str(row['all_finite']):>7} "
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

    print("\nPer-case reference diagnostics")
    for result in case_results:
        case = result["case"]
        ref = result["reference_crosscheck"]
        print(
            f"{case['name']:<28} beta=({case['beta_0']:+.2f},{case['beta_inf']:+.2f}) "
            f"ra/Re={case['r_a_over_re']:.3g} rs/Re={case['rs_over_re']:.3g} "
            f"ref_rel={float(ref['max_rel']):.3e} "
            f"near_zero={float(ref['near_zero_fraction']):.2f} "
            f"warn={result['reference_warning']}"
        )

    payload = {
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "n_radii": n_radii,
            "R_over_re_min": 0.005,
            "R_over_re_max": 10.0,
            "u_max": u_max,
            "n_kernel": n_kernel,
            "n_ref_outer": n_ref_outer,
            "n_ref_abel": n_ref_abel,
        },
        "worst_reference_crosscheck": worst_ref_cross,
        "reference_warning_cases": warned_cases,
        "aggregate_rows": aggregate_rows,
        "best_at_tolerance": best,
        "case_results": case_results,
    }
    print("\nJSON")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
