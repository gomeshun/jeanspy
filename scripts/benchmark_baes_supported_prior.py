from __future__ import annotations

import os
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
    eta: float


def _cases() -> list[Case]:
    beta_pairs = [
        ("radial-max", -9.0, 0.98),
        ("radial-strong", -5.0, 0.98),
        ("radial-moderate", -1.0, 0.98),
        ("radial-inneriso", 0.0, 0.98),
        ("tangential-max", 0.0, -9.0),
        ("tangential-strong", -1.0, -9.0),
    ]
    etas = (0.1, 1.0, 4.0, 10.0)
    ra_ratios = (0.005, 1.0, 50.0)
    out: list[Case] = []
    for label, beta0, betainf in beta_pairs:
        for eta in etas:
            for ra_ratio in ra_ratios:
                out.append(
                    Case(
                        f"{label}-eta{eta:g}-ra{ra_ratio:g}",
                        beta0,
                        betainf,
                        ra_ratio,
                        eta,
                    )
                )
    return out


def max_rel(a, b) -> float:
    import numpy as np

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if not (np.isfinite(aa).all() and np.isfinite(bb).all()):
        return float("inf")
    scale = max(float(np.max(np.abs(bb))), 1e-12)
    floor = max(scale * 1e-10, 1e-12)
    return float(np.max(np.abs(aa - bb) / np.maximum(np.abs(bb), floor)))


def main() -> None:
    import numpy as np
    import jax
    import jax.numpy as jnp

    import jeanspy.model_numpyro as mn
    from jeanspy.model_numpyro import NFWModel, PlummerModel

    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("Run with JAX_ENABLE_X64=true")

    cases = _cases()
    re_pc = 220.0
    R_over_re = np.geomspace(0.005, 10.0, 28)
    stellar = PlummerModel()
    dm = NFWModel()

    def stable_kernel(u, R, params, *, n_kernel):
        u_arr, r_arr = jnp.broadcast_arrays(jnp.asarray(u), jnp.asarray(R))
        dtype = jnp.result_type(
            u_arr,
            r_arr,
            params["beta_0"],
            params["beta_inf"],
            params["r_a"],
            params["eta"],
        )
        one = jnp.asarray(1.0, dtype=dtype)
        eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)

        u_in = u_arr.astype(dtype)
        u_safe = jnp.maximum(u_in, one + eps)
        R_safe = jnp.maximum(r_arr.astype(dtype), tiny)
        beta0 = jnp.asarray(params["beta_0"], dtype=dtype)
        q = jnp.asarray(params["beta_inf"], dtype=dtype) - beta0
        ra = jnp.maximum(jnp.asarray(params["r_a"], dtype=dtype), tiny)
        eta = jnp.maximum(jnp.asarray(params["eta"], dtype=dtype), tiny)

        s_max = jnp.arccosh(u_safe)
        nodes_np, weights_np = mn._gauss_legendre_01(int(n_kernel))
        nodes = jnp.asarray(nodes_np, dtype=dtype)
        weights = jnp.asarray(weights_np, dtype=dtype)
        s = s_max[..., None] * nodes
        uint = jnp.cosh(s)

        log_R_over_ra = jnp.log(R_safe / ra)
        log_rint_over_ra = log_R_over_ra[..., None] + jnp.log(uint)
        log_rs_over_ra = log_R_over_ra + jnp.log(u_safe)
        ell_int = eta * log_rint_over_ra
        ell_s = eta * log_rs_over_ra

        beta_int = beta0 + q * jax.nn.sigmoid(ell_int)
        log_ratio = (
            2.0 * beta0 * (jnp.log(u_safe)[..., None] - jnp.log(uint))
            + (2.0 * q / eta)
            * (jax.nn.softplus(ell_s)[..., None] - jax.nn.softplus(ell_int))
        )
        log_ratio = jnp.clip(log_ratio, min=-80.0, max=80.0)
        integrand = uint * (one - beta_int / (uint * uint)) * jnp.exp(log_ratio)
        inner = s_max * jnp.sum(weights * integrand, axis=-1)
        out = inner / u_safe
        out = jnp.where(u_in <= one, jnp.zeros_like(out), out)
        return jnp.nan_to_num(out, nan=0.0, neginf=0.0, posinf=1e12)

    def projection(R, params, *, n_u, n_kernel, u_max):
        dtype = R.dtype
        x = jnp.linspace(
            jnp.asarray(0.0, dtype=dtype),
            jnp.sqrt(jnp.log(jnp.asarray(u_max, dtype=dtype))),
            int(n_u),
        )
        u = jnp.exp(x * x)
        R2d = R[:, None]
        u2d = u[None, :]
        r = R2d * u2d
        nu3 = stellar.density_3d(r, re_pc=params["re_pc"])
        sigma2 = stellar.density_2d(R2d, re_pc=params["re_pc"])
        mass = dm.enclosed_mass(r, method="analytic", params=params)
        grav = (mn.GMsun_m3s2 * mass / mn.PARSEC_M) * 1e-6
        K = stable_kernel(u2d, R2d, params, n_kernel=n_kernel)
        integrand = 2.0 * K * (nu3 / sigma2) * grav * (2.0 * x)[None, :]
        return mn._simpson_uniform_last_axis(integrand, x[1] - x[0])

    orders = (16, 32, 64, 128)
    for dtype_name, dtype in (("float64", jnp.float64), ("float32", jnp.float32)):
        worst_inner = {order: (0.0, "") for order in orders}
        worst_prod = (0.0, "")
        for case in cases:
            raw = {
                "re_pc": re_pc,
                "rs_pc": 5.0 * re_pc,
                "rhos_Msunpc3": 7.5e-3,
                "r_t_pc": 40.0 * re_pc,
                "beta_0": case.beta_0,
                "beta_inf": case.beta_inf,
                "r_a": case.r_a_over_re * re_pc,
                "eta": case.eta,
                "vmem_kms": 0.0,
            }
            params = {k: jnp.asarray(v, dtype=dtype) for k, v in raw.items()}
            params64 = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in raw.items()}
            R = jnp.asarray(R_over_re * re_pc, dtype=dtype)
            R64 = jnp.asarray(R_over_re * re_pc, dtype=jnp.float64)

            # Same outer grid: isolate inner-kernel quadrature order.
            ref_inner = projection(R64, params64, n_u=256, n_kernel=256, u_max=2e4)
            for order in orders:
                candidate = projection(R, params, n_u=256, n_kernel=order, u_max=2e4)
                err = max_rel(candidate, ref_inner)
                if err > worst_inner[order][0]:
                    worst_inner[order] = (err, case.name)

            # Candidate production setup versus a much higher-resolution reference.
            ref_full = projection(R64, params64, n_u=1024, n_kernel=256, u_max=1e5)
            prod = projection(R, params, n_u=64, n_kernel=32, u_max=5000.0)
            err_prod = max_rel(prod, ref_full)
            if err_prod > worst_prod[0]:
                worst_prod = (err_prod, case.name)

        print(f"\nBaes supported-prior stress: {dtype_name}")
        print(f"cases={len(cases)}, eta in [0.1,10], beta edges to -9/+0.98, r_a/Re in [0.005,50]")
        print("inner-order-only comparison: n_u=256, u_max=2e4, reference n_kernel=256")
        for order in orders:
            err, name = worst_inner[order]
            print(f"  n_kernel={order:3d}: worst_rel={err:.3e}  case={name}")
        print("production candidate: n_u=64, n_kernel=32, u_max=5000 vs n_u=1024, n_kernel=256, u_max=1e5")
        print(f"  worst_rel={worst_prod[0]:.3e}  case={worst_prod[1]}")


if __name__ == "__main__":
    main()
