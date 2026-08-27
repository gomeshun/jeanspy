from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from typing import Iterable

# The benchmark needs float64 references.  Float32 candidates are still evaluated
# with float32 arrays in the same process so they can be compared directly.
os.environ.setdefault("JEANSPY_JAX_PLATFORM", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


ACCURACY_TARGET = 1.0e-3
RELATIVE_FLOOR_FRACTION = 1.0e-9
ABSOLUTE_FLOOR = 1.0e-12
REFERENCE_N_U = 1024
REFERENCE_N_KERNEL = 256
REFERENCE_U_MAX = 1.0e5
CPU_N_U = 128
GPU_X32_N_U = 1024
DEFAULT_U_MAX = 1.0e4


@dataclass(frozen=True)
class Case:
    name: str
    anisotropy: str
    rs_over_re: float
    r_t_over_re: float = 40.0
    beta_ani: float = 0.0
    beta_0: float = 0.0
    beta_inf: float = 0.0
    r_a_over_re: float = 1.0
    eta: float = 2.0


@dataclass(frozen=True)
class Profile:
    name: str
    dtype_name: str
    n_u: int
    u_max: float
    constant_n_kernel: int
    baes_n_kernel: int
    public_cpu_defaults: bool


PROFILES = (
    Profile("cpu-float64", "float64", CPU_N_U, DEFAULT_U_MAX, 32, 32, True),
    Profile("cpu-float32", "float32", CPU_N_U, DEFAULT_U_MAX, 32, 32, True),
    # GitHub-hosted CI has no GPU.  This profile reproduces the numerical grid and
    # float32 arithmetic of the GPU-oriented defaults on CPU; its runtime is not a
    # GPU performance measurement.
    Profile(
        "gpu-float32-numerical-proxy",
        "float32",
        GPU_X32_N_U,
        DEFAULT_U_MAX,
        64,
        32,
        False,
    ),
)


CI_CASES = (
    Case("constant-near-radial-broad", "constant", 100.0, beta_ani=0.98),
    Case("constant-tangential", "constant", 5.0, beta_ani=-9.0),
    Case("om-radial-broad", "om", 100.0, r_a_over_re=0.05),
    Case("om-outer-transition", "om", 5.0, r_a_over_re=50.0),
    Case(
        "baes-prior-edge-broad",
        "baes",
        100.0,
        beta_0=-9.0,
        beta_inf=0.98,
        r_a_over_re=1.0,
        eta=4.0,
    ),
    Case(
        "baes-rapid-radializing",
        "baes",
        5.0,
        beta_0=-9.0,
        beta_inf=0.98,
        r_a_over_re=0.005,
        eta=10.0,
    ),
    Case(
        "baes-outer-tangential",
        "baes",
        5.0,
        beta_0=0.0,
        beta_inf=-9.0,
        r_a_over_re=50.0,
        eta=0.1,
    ),
    Case(
        "baes-fiducial-broad",
        "baes",
        100.0,
        beta_0=-0.5,
        beta_inf=0.65,
        r_a_over_re=1.36,
        eta=2.0,
    ),
)


def _full_cases() -> list[Case]:
    out: list[Case] = []
    rs_ratios = (0.05, 5.0, 100.0)
    for rs_ratio in rs_ratios:
        for beta in (-9.0, -5.0, -1.0, 0.0, 0.5, 0.98):
            out.append(
                Case(
                    f"constant-rs{rs_ratio:g}-beta{beta:g}",
                    "constant",
                    rs_ratio,
                    beta_ani=beta,
                )
            )
        for ra_ratio in (0.005, 0.05, 1.0, 50.0):
            out.append(
                Case(
                    f"om-rs{rs_ratio:g}-ra{ra_ratio:g}",
                    "om",
                    rs_ratio,
                    r_a_over_re=ra_ratio,
                )
            )

    beta_pairs = (
        ("radial-max", -9.0, 0.98),
        ("radial-strong", -5.0, 0.98),
        ("radial-moderate", -1.0, 0.98),
        ("radial-inneriso", 0.0, 0.98),
        ("tangential-max", 0.0, -9.0),
        ("tangential-strong", -1.0, -9.0),
    )
    for rs_ratio in rs_ratios:
        for label, beta_0, beta_inf in beta_pairs:
            for eta in (0.1, 1.0, 4.0, 10.0):
                for ra_ratio in (0.005, 1.0, 50.0):
                    out.append(
                        Case(
                            f"baes-{label}-rs{rs_ratio:g}-eta{eta:g}-ra{ra_ratio:g}",
                            "baes",
                            rs_ratio,
                            beta_0=beta_0,
                            beta_inf=beta_inf,
                            r_a_over_re=ra_ratio,
                            eta=eta,
                        )
                    )
    return out


def _max_rel(candidate, reference) -> float:
    import numpy as np

    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if not (np.isfinite(candidate).all() and np.isfinite(reference).all()):
        return float("inf")
    global_scale = max(float(np.max(np.abs(reference))), ABSOLUTE_FLOOR)
    floor = max(global_scale * RELATIVE_FLOOR_FRACTION, ABSOLUTE_FLOOR)
    denom = np.maximum(np.abs(reference), floor)
    return float(np.max(np.abs(candidate - reference) / denom))


def _make_model(case: Case):
    from jeanspy.model_numpyro import (
        BaesAnisotropyModel,
        ConstantAnisotropyModel,
        DSphModel,
        NFWModel,
        OsipkovMerrittModel,
        PlummerModel,
    )

    if case.anisotropy == "constant":
        ani = ConstantAnisotropyModel()
    elif case.anisotropy == "om":
        ani = OsipkovMerrittModel()
    elif case.anisotropy == "baes":
        ani = BaesAnisotropyModel()
    else:  # pragma: no cover - Case values are defined in this file.
        raise ValueError(case.anisotropy)

    return DSphModel(
        submodels={
            "StellarModel": PlummerModel(),
            "DMModel": NFWModel(),
            "AnisotropyModel": ani,
        }
    )


def _params(case: Case, dtype, *, re_pc: float):
    import jax.numpy as jnp

    raw = {
        "re_pc": re_pc,
        "rs_pc": case.rs_over_re * re_pc,
        "rhos_Msunpc3": 7.5e-3,
        "r_t_pc": case.r_t_over_re * re_pc,
        "vmem_kms": 0.0,
    }
    if case.anisotropy == "constant":
        raw["beta_ani"] = case.beta_ani
    elif case.anisotropy == "om":
        raw["r_a"] = case.r_a_over_re * re_pc
    else:
        raw.update(
            {
                "beta_0": case.beta_0,
                "beta_inf": case.beta_inf,
                "r_a": case.r_a_over_re * re_pc,
                "eta": case.eta,
            }
        )
    return {key: jnp.asarray(value, dtype=dtype) for key, value in raw.items()}


def _n_kernel_for(profile: Profile, case: Case) -> int | None:
    if case.anisotropy == "constant":
        return profile.constant_n_kernel
    if case.anisotropy == "baes":
        return profile.baes_n_kernel
    return None


def _evaluate_candidate(dsph, R, params, case: Case, profile: Profile):
    kwargs = {
        "backend": "kernel",
        "dm_mass_method": "analytic",
        "jit": True,
    }
    n_kernel = _n_kernel_for(profile, case)
    if n_kernel is not None:
        kwargs["n_kernel"] = n_kernel

    if profile.public_cpu_defaults:
        # Deliberately omit n_u/u_max: the CI gate is tied to the public CPU
        # defaults rather than merely reproducing a hard-coded benchmark grid.
        return dsph.sigmalos2(R, params=params, **kwargs)

    return dsph.sigmalos2(
        R,
        params=params,
        n_u=profile.n_u,
        u_max=profile.u_max,
        **kwargs,
    )


def _evaluate_reference(dsph, R64, params64, case: Case):
    kwargs = {
        "backend": "kernel",
        "n_u": REFERENCE_N_U,
        "u_max": REFERENCE_U_MAX,
        "dm_mass_method": "analytic",
        "jit": True,
    }
    if case.anisotropy in {"constant", "baes"}:
        kwargs["n_kernel"] = REFERENCE_N_KERNEL
    return dsph.sigmalos2(R64, params=params64, **kwargs)


def _time_hot(jax, func, repeats: int = 3) -> float:
    jax.block_until_ready(func())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(func())
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples))


def _check_cpu_defaults() -> None:
    from jeanspy.model_numpyro import get_runtime_config

    config = get_runtime_config()
    if config["jax_backend_active"] != "cpu":
        raise RuntimeError("This benchmark is intended to run on the CPU reference host")
    if config["sigmalos2_n_u_default"] != CPU_N_U:
        raise AssertionError(
            "CPU sigmalos2 n_u default changed: "
            f"expected {CPU_N_U}, got {config['sigmalos2_n_u_default']}"
        )
    if config["sigmalos2_u_max_default"] != DEFAULT_U_MAX:
        raise AssertionError(
            "CPU sigmalos2 u_max default changed: "
            f"expected {DEFAULT_U_MAX:g}, got {config['sigmalos2_u_max_default']:g}"
        )


def run(cases: Iterable[Case], *, enforce: bool) -> None:
    import numpy as np
    import jax
    import jax.numpy as jnp

    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("The benchmark requires JAX_ENABLE_X64=true for references")

    _check_cpu_defaults()
    cases = tuple(cases)
    re_pc = 220.0
    R_over_re = np.geomspace(0.005, 10.0, 28)
    R64 = jnp.asarray(R_over_re * re_pc, dtype=jnp.float64)

    references = {}
    models = {}
    for case in cases:
        dsph = _make_model(case)
        models[case.name] = dsph
        params64 = _params(case, jnp.float64, re_pc=re_pc)
        references[case.name] = jax.block_until_ready(
            _evaluate_reference(dsph, R64, params64, case)
        )

    print("sigmalos2 kernel numerical-accuracy contract")
    print(f"target: max relative error <= {ACCURACY_TARGET:.1e}")
    print(
        "metric floor: max(|reference|, "
        f"{RELATIVE_FLOOR_FRACTION:.0e} * max|reference|, {ABSOLUTE_FLOOR:.0e})"
    )
    print(
        "reference: CPU float64 kernel, "
        f"n_u={REFERENCE_N_U}, n_kernel={REFERENCE_N_KERNEL}, u_max={REFERENCE_U_MAX:g}"
    )
    print(f"cases={len(cases)}, R/Re=[0.005,10]")

    failures = []
    for profile in PROFILES:
        dtype = jnp.float64 if profile.dtype_name == "float64" else jnp.float32
        R = jnp.asarray(R_over_re * re_pc, dtype=dtype)
        errors = []
        runtimes = []
        for case in cases:
            dsph = models[case.name]
            params = _params(case, dtype, re_pc=re_pc)

            def candidate():
                return _evaluate_candidate(dsph, R, params, case, profile)

            value = jax.block_until_ready(candidate())
            error = _max_rel(value, references[case.name])
            errors.append((error, case.name))
            runtimes.append(_time_hot(jax, candidate))

        worst_error, worst_case = max(errors, key=lambda item: item[0])
        print(
            f"{profile.name:29s} worst_rel={worst_error:.3e} "
            f"median_hot_ms={1e3 * statistics.median(runtimes):.3f} "
            f"worst_case={worst_case}"
        )
        if worst_error > ACCURACY_TARGET:
            failures.append((profile.name, worst_error, worst_case))

    if enforce and failures:
        details = "; ".join(
            f"{name}: {error:.3e} ({case})" for name, error, case in failures
        )
        raise SystemExit(
            f"sigmalos2 accuracy contract failed ({ACCURACY_TARGET:.1e}): {details}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the documented sigmalos2 kernel accuracy contract."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full supported-domain stress matrix instead of the compact CI set.",
    )
    parser.add_argument(
        "--no-enforce",
        action="store_true",
        help="Report errors without exiting non-zero when the target is exceeded.",
    )
    args = parser.parse_args()
    run(_full_cases() if args.full else CI_CASES, enforce=not args.no_enforce)


if __name__ == "__main__":
    main()
