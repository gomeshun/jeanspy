from __future__ import annotations

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


def sync(jax, value):
    return jax.block_until_ready(value)


def max_rel(a, b) -> float:
    import numpy as np

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    scale = max(float(np.max(np.abs(bb))), 1e-12)
    floor = max(scale * 1e-9, 1e-12)
    return float(np.max(np.abs(aa - bb) / np.maximum(np.abs(bb), floor)))


def median_runtime(jax, fn: Callable[[dict[str, Any]], Any], params, repeats=3) -> float:
    sync(jax, fn(params))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sync(jax, fn(params))
        times.append(time.perf_counter() - t0)
    return float(statistics.median(times))


def main() -> None:
    import numpy as np
    import jax
    import jax.numpy as jnp

    import jeanspy.model_numpyro as mn
    from jeanspy.baes_eta2 import BaesEta2AnisotropyModel
    from jeanspy.model_numpyro import BaesAnisotropyModel, NFWModel, PlummerModel

    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("This benchmark requires JAX float64.")

    dtype = jnp.float64
    re_pc = 220.0
    R_over_re = np.geomspace(0.005, 10.0, 40)
    R = jnp.asarray(R_over_re * re_pc, dtype=dtype)

    stellar = PlummerModel()
    dm = NFWModel()
    eta2 = BaesEta2AnisotropyModel()
    generic = BaesAnisotropyModel()

    def params_for(case: Case, *, generic_model: bool) -> dict[str, Any]:
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
        if generic_model:
            raw["eta"] = 2.0
        return {key: jnp.asarray(value, dtype=dtype) for key, value in raw.items()}

    def projection_fn(
        ani,
        *,
        n_u: int,
        n_kernel: int,
        u_max: float,
        transform: str = "sqrtlog",
        u_min_eps: float = 1e-6,
    ):
        if transform == "sqrtlog":
            x_max = jnp.sqrt(jnp.log(jnp.asarray(u_max, dtype=dtype)))
            x = jnp.linspace(jnp.asarray(0.0, dtype=dtype), x_max, n_u)
            u = jnp.exp(x * x)
            jac = 2.0 * x
        elif transform == "log":
            x = jnp.linspace(
                jnp.log1p(jnp.asarray(u_min_eps, dtype=dtype)),
                jnp.log(jnp.asarray(u_max, dtype=dtype)),
                n_u,
            )
            u = jnp.exp(x)
            jac = jnp.ones_like(x)
        else:
            raise ValueError(transform)

        u2d = u[None, :]
        R2d = R[:, None]
        h = x[1] - x[0]

        @jax.jit
        def _eval(params):
            r = R2d * u2d
            nu3 = stellar.density_3d(r, re_pc=params["re_pc"])
            sigma2 = stellar.density_2d(R2d, re_pc=params["re_pc"])
            mass = dm.enclosed_mass(r, method="analytic", params=params)
            grav = (mn.GMsun_m3s2 * mass / mn.PARSEC_M) * 1e-6
            K = ani.kernel(u2d, R2d, params=params, n_kernel=n_kernel)
            # In log(u), K/u * du = K dx.  In sqrt(log(u)),
            # log(u)=x^2 and therefore K/u * du = 2x K dx.
            integrand = 2.0 * K * (nu3 / sigma2) * grav * jac[None, :]
            return mn._simpson_uniform_last_axis(integrand, h)

        return _eval

    # Independent robust reference: generic Baes integrates the anisotropy
    # kernel in s=arccosh(u), which remains well resolved for beta_inf -> 1.
    # Use a large outer cutoff to make small-R tail truncation negligible.
    ref_mid_fn = projection_fn(
        generic, n_u=512, n_kernel=128, u_max=1e5, transform="sqrtlog"
    )
    ref_hi_fn = projection_fn(
        generic, n_u=1024, n_kernel=256, u_max=1e5, transform="sqrtlog"
    )

    case_data = []
    worst_ref_check = 0.0
    for case in CASES:
        pg = params_for(case, generic_model=True)
        pe = params_for(case, generic_model=False)
        ref_mid = sync(jax, ref_mid_fn(pg))
        ref_hi = sync(jax, ref_hi_fn(pg))
        ref_check = max_rel(ref_mid, ref_hi)
        worst_ref_check = max(worst_ref_check, ref_check)
        case_data.append((case, pg, pe, ref_hi, ref_check))

    print("eta=2 outer-transform adversarial MCMC stress benchmark (CPU float64)")
    print(
        f"cases={len(CASES)}, R/Re=[0.005,10], "
        f"robust generic-reference check={worst_ref_check:.3e}"
    )
    print("reference checks by case:")
    for case, _, _, _, ref_check in case_data:
        print(f"  {case.name:<28} {ref_check:.3e}")

    candidate_specs = [
        (32, 32),
        (64, 32),
        (64, 64),
        (64, 128),
        (128, 32),
        (128, 64),
        (128, 128),
        (256, 64),
        (256, 128),
    ]

    print("\neta2 candidates with sqrt(log u), u_max=2e4")
    print("n_u  n_kernel  worst_rel  median_case_ms  max_case_ms  worst_case")
    for n_u, n_kernel in candidate_specs:
        fn = projection_fn(
            eta2, n_u=n_u, n_kernel=n_kernel, u_max=2e4, transform="sqrtlog"
        )
        errors = []
        runtimes = []
        for case, _, pe, ref_hi, _ in case_data:
            value = sync(jax, fn(pe))
            errors.append((max_rel(value, ref_hi), case.name))
            runtimes.append(median_runtime(jax, fn, pe))
        worst_err, worst_name = max(errors, key=lambda item: item[0])
        print(
            f"{n_u:>3d} {n_kernel:>9d} {worst_err:>10.3e} "
            f"{1e3*statistics.median(runtimes):>15.3f} "
            f"{1e3*max(runtimes):>12.3f}  {worst_name}"
        )

    # Compare against using the robust generic kernel with the same improved
    # outer transform.  This shows how much speed is specifically due to the
    # eta=2 inner evaluator rather than the sqrt(log u) outer transformation.
    generic_specs = [(32, 32), (64, 32), (64, 64), (128, 32), (128, 64)]
    print("\ngeneric Baes with sqrt(log u), u_max=2e4")
    print("n_u  n_kernel  worst_rel  median_case_ms  max_case_ms  worst_case")
    for n_u, n_kernel in generic_specs:
        fn = projection_fn(
            generic, n_u=n_u, n_kernel=n_kernel, u_max=2e4, transform="sqrtlog"
        )
        errors = []
        runtimes = []
        for case, pg, _, ref_hi, _ in case_data:
            value = sync(jax, fn(pg))
            errors.append((max_rel(value, ref_hi), case.name))
            runtimes.append(median_runtime(jax, fn, pg))
        worst_err, worst_name = max(errors, key=lambda item: item[0])
        print(
            f"{n_u:>3d} {n_kernel:>9d} {worst_err:>10.3e} "
            f"{1e3*statistics.median(runtimes):>15.3f} "
            f"{1e3*max(runtimes):>12.3f}  {worst_name}"
        )

    # Directly compare against the current production outer grid.  Keep the
    # same generic Baes inner kernel and u_max so the only change is the outer
    # coordinate.  n_kernel=32 is already sufficient for the robust generic
    # arccosh(u) inner quadrature over this adversarial set.
    print("\ngeneric Baes with current log(u), u_max=2e4")
    print("n_u  n_kernel  worst_rel  median_case_ms  max_case_ms  worst_case")
    for n_u in (64, 128, 256, 512, 1024):
        fn = projection_fn(
            generic, n_u=n_u, n_kernel=32, u_max=2e4, transform="log"
        )
        errors = []
        runtimes = []
        for case, pg, _, ref_hi, _ in case_data:
            value = sync(jax, fn(pg))
            errors.append((max_rel(value, ref_hi), case.name))
            runtimes.append(median_runtime(jax, fn, pg))
        worst_err, worst_name = max(errors, key=lambda item: item[0])
        print(
            f"{n_u:>4d} {32:>9d} {worst_err:>10.3e} "
            f"{1e3*statistics.median(runtimes):>15.3f} "
            f"{1e3*max(runtimes):>12.3f}  {worst_name}"
        )

    # Also record the current CPU defaults (n_u=256, u_max=2e3).  This is not
    # an apples-to-apples transform comparison because the cutoff is smaller,
    # but it quantifies the accuracy users currently obtain in this stress set.
    current_default = projection_fn(
        generic, n_u=256, n_kernel=32, u_max=2e3, transform="log"
    )
    default_errors = []
    default_runtimes = []
    for case, pg, _, ref_hi, _ in case_data:
        value = sync(jax, current_default(pg))
        default_errors.append((max_rel(value, ref_hi), case.name))
        default_runtimes.append(median_runtime(jax, current_default, pg))
    default_err, default_name = max(default_errors, key=lambda item: item[0])
    print("\ncurrent CPU-like log(u) configuration")
    print(
        f"n_u=256 n_kernel=32 u_max=2e3: worst_rel={default_err:.3e}, "
        f"median_case_ms={1e3*statistics.median(default_runtimes):.3f}, "
        f"worst_case={default_name}"
    )


if __name__ == "__main__":
    main()
