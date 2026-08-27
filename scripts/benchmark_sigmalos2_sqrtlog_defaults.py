from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

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


# Deliberately numerical/MCMC-adversarial rather than a physical-prior catalogue.
CASES = [
    Case("fiducial", -0.5, 0.65, 1.36, 5.0),
    Case("inner-transition", -1.0, 0.8, 0.25, 5.0),
    Case("outer-transition", 0.4, 0.8, 5.0, 5.0),
    Case("moderate-tangential", -1.5, -0.2, 1.0, 5.0),
    Case("constant-radial", 0.7, 0.7, 1.0, 2.0),
    Case("near-radial-constant", 0.98, 0.98, 1.0, 5.0),
    Case("extreme-tangential-constant", -5.0, -5.0, 1.0, 5.0),
    Case("rapid-radializing", -5.0, 0.98, 0.01, 5.0),
    Case("rapid-tangentializing", 0.98, -5.0, 0.01, 5.0),
    Case("tiny-ra", -0.5, 0.98, 1e-3, 5.0),
    Case("huge-ra", 0.98, -0.5, 1e3, 5.0),
    Case("compact-halo", -0.5, 0.65, 1.36, 0.05),
    Case("broad-halo", -0.5, 0.65, 1.36, 12.0),
    Case("ultra-broad-halo", -0.5, 0.65, 1.36, 100.0),
]


def _max_rel(a, b) -> float:
    import numpy as np

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    scale = max(float(np.max(np.abs(bb))), 1e-12)
    floor = max(scale * 1e-9, 1e-12)
    return float(np.max(np.abs(aa - bb) / np.maximum(np.abs(bb), floor)))


def main() -> None:
    import numpy as np
    import jax
    import jax.numpy as jnp

    from jeanspy.model_numpyro import (
        BaesAnisotropyModel,
        DSphModel,
        NFWModel,
        PlummerModel,
    )

    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("Run this benchmark with JAX_ENABLE_X64=true")

    re_pc = 220.0
    R_over_re = np.geomspace(0.005, 10.0, 40)

    dsph = DSphModel(
        submodels={
            "StellarModel": PlummerModel(),
            "DMModel": NFWModel(),
            "AnisotropyModel": BaesAnisotropyModel(),
        }
    )

    def params_for(case: Case, dtype):
        raw = {
            "re_pc": re_pc,
            "rs_pc": case.rs_over_re * re_pc,
            "rhos_Msunpc3": 7.5e-3,
            "r_t_pc": 40.0 * re_pc,
            "beta_0": case.beta_0,
            "beta_inf": case.beta_inf,
            "r_a": case.r_a_over_re * re_pc,
            "eta": 2.0,
            "vmem_kms": 0.0,
        }
        return {key: jnp.asarray(value, dtype=dtype) for key, value in raw.items()}

    R64 = jnp.asarray(R_over_re * re_pc, dtype=jnp.float64)
    refs = {}
    # Robust public-API reference. The earlier convergence study showed the
    # generic arccosh(u) BAES kernel is stable in the beta->1, large-u regime.
    for case in CASES:
        params64 = params_for(case, jnp.float64)
        refs[case.name] = jax.block_until_ready(
            dsph.sigmalos2(
                R64,
                params=params64,
                backend="kernel",
                n_u=1024,
                n_kernel=256,
                u_max=1.0e5,
                kernel_outer_transform="sqrtlog",
                dm_mass_method="analytic",
                jit=True,
            )
        )

    specs = [
        (64, 32, 2.0e3),
        (128, 32, 2.0e3),
        (256, 32, 2.0e3),
        (64, 32, 5.0e3),
        (128, 32, 5.0e3),
        (64, 64, 5.0e3),
        (64, 32, 2.0e4),
        (128, 32, 2.0e4),
    ]

    print("Public DSphModel.sigmalos2 sqrtlog default study")
    print("reference: float64 n_u=1024 n_kernel=256 u_max=1e5")
    print("cases=14, R/Re=[0.005,10]")

    for dtype_name, dtype in (("float64", jnp.float64), ("float32", jnp.float32)):
        R = jnp.asarray(R_over_re * re_pc, dtype=dtype)
        print(f"\n{dtype_name}")
        print("n_u  n_kernel  u_max    worst_rel   median_hot_ms   worst_case")
        for n_u, n_kernel, u_max in specs:
            errors = []
            runtimes = []
            for case in CASES:
                params = params_for(case, dtype)
                # Warm/compile for this shape+dtype+static numerical configuration.
                value = jax.block_until_ready(
                    dsph.sigmalos2(
                        R,
                        params=params,
                        backend="kernel",
                        n_u=n_u,
                        n_kernel=n_kernel,
                        u_max=u_max,
                        kernel_outer_transform="sqrtlog",
                        dm_mass_method="analytic",
                        jit=True,
                    )
                )
                errors.append((_max_rel(value, refs[case.name]), case.name))

                samples = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    value = dsph.sigmalos2(
                        R,
                        params=params,
                        backend="kernel",
                        n_u=n_u,
                        n_kernel=n_kernel,
                        u_max=u_max,
                        kernel_outer_transform="sqrtlog",
                        dm_mass_method="analytic",
                        jit=True,
                    )
                    jax.block_until_ready(value)
                    samples.append(time.perf_counter() - t0)
                runtimes.append(statistics.median(samples))

            worst_err, worst_case = max(errors, key=lambda item: item[0])
            print(
                f"{n_u:>3d} {n_kernel:>9d} {u_max:>7.0f} "
                f"{worst_err:>11.3e} {1e3*statistics.median(runtimes):>15.3f} "
                f"{worst_case}"
            )


if __name__ == "__main__":
    main()
