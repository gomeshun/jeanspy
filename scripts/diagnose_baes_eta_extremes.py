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


CASES = [
    Case("eta005-radializing", -1.0, 0.95, 1.0, 0.05),
    Case("eta010-radializing", -1.0, 0.95, 1.0, 0.10),
    Case("eta025-radializing", -1.0, 0.95, 1.0, 0.25),
    Case("eta050-radializing", -1.0, 0.95, 1.0, 0.50),
    Case("eta100-radializing", -1.0, 0.95, 1.0, 1.00),
    Case("eta200-radializing", -1.0, 0.95, 1.0, 2.00),
    Case("eta400-radializing", -1.0, 0.95, 1.0, 4.00),
    Case("eta1000-radializing", -1.0, 0.95, 1.0, 10.0),
    Case("eta2000-radializing", -1.0, 0.95, 1.0, 20.0),
    Case("eta005-tangentializing", 0.95, -5.0, 1.0, 0.05),
    Case("eta020-tangentializing", 0.95, -5.0, 0.01, 0.20),
    Case("eta1000-tangentializing", 0.95, -5.0, 0.01, 10.0),
    Case("eta2000-tangentializing", 0.95, -5.0, 0.01, 20.0),
    Case("eta2000-tiny-ra", -5.0, 0.98, 1e-3, 20.0),
    Case("eta2000-huge-ra", 0.98, -5.0, 1e3, 20.0),
]


def max_rel(a, b) -> float:
    import numpy as np

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(aa) & np.isfinite(bb)
    if not finite.all():
        return float("inf")
    scale = max(float(np.max(np.abs(bb))), 1e-12)
    floor = max(scale * 1e-10, 1e-12)
    return float(np.max(np.abs(aa - bb) / np.maximum(np.abs(bb), floor)))


def main() -> None:
    import numpy as np
    import jax
    import jax.numpy as jnp

    import jeanspy.model_numpyro as mn
    from jeanspy.model_numpyro import BaesAnisotropyModel, NFWModel, PlummerModel

    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("Run with JAX_ENABLE_X64=true")

    re_pc = 220.0
    R_over_re = np.geomspace(0.005, 10.0, 28)
    u_probes = np.asarray([1.0 + 1e-6, 1.01, 1.1, 2, 10, 100, 1e3, 1e4, 1e5])
    generic = BaesAnisotropyModel()
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
        return jnp.where(u_in <= one, jnp.zeros_like(out), out)

    def projection(ani_kernel, R, params, *, n_u, n_kernel, u_max):
        dtype = R.dtype
        x = jnp.linspace(
            jnp.asarray(0.0, dtype=dtype),
            jnp.sqrt(jnp.log(jnp.asarray(u_max, dtype=dtype))),
            n_u,
        )
        u = jnp.exp(x * x)
        R2d = R[:, None]
        u2d = u[None, :]
        r = R2d * u2d
        nu3 = stellar.density_3d(r, re_pc=params["re_pc"])
        sigma2 = stellar.density_2d(R2d, re_pc=params["re_pc"])
        mass = dm.enclosed_mass(r, method="analytic", params=params)
        grav = (mn.GMsun_m3s2 * mass / mn.PARSEC_M) * 1e-6
        K = ani_kernel(u2d, R2d, params, n_kernel=n_kernel)
        integrand = 2.0 * K * (nu3 / sigma2) * grav * (2.0 * x)[None, :]
        return mn._simpson_uniform_last_axis(integrand, x[1] - x[0])

    def current_kernel(u, R, params, *, n_kernel):
        return generic.kernel(u, R, params=params, n_kernel=n_kernel)

    for dtype_name, dtype in (("float64", jnp.float64), ("float32", jnp.float32)):
        print(f"\nGeneric Baes eta-extreme diagnostic: {dtype_name}")
        print(
            "case                         current64_ref  stable32_ref  "
            "proj_cur32  proj_stable32  finite_cur"
        )
        for case in CASES:
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

            R_probe64 = jnp.asarray([0.005, 0.1, 1.0, 10.0], dtype=jnp.float64)[:, None] * re_pc
            u_probe64 = jnp.asarray(u_probes, dtype=jnp.float64)[None, :]
            ref_kernel = stable_kernel(u_probe64, R_probe64, params64, n_kernel=512)

            R_probe = jnp.asarray([0.005, 0.1, 1.0, 10.0], dtype=dtype)[:, None] * re_pc
            u_probe = jnp.asarray(u_probes, dtype=dtype)[None, :]
            current = current_kernel(u_probe, R_probe, params, n_kernel=32)
            stable32 = stable_kernel(u_probe, R_probe, params, n_kernel=32)

            R = jnp.asarray(R_over_re * re_pc, dtype=dtype)
            ref_proj = projection(
                stable_kernel,
                jnp.asarray(R_over_re * re_pc, dtype=jnp.float64),
                params64,
                n_u=512,
                n_kernel=256,
                u_max=2e4,
            )
            proj_current = projection(
                current_kernel, R, params, n_u=64, n_kernel=32, u_max=2e4
            )
            proj_stable = projection(
                stable_kernel, R, params, n_u=64, n_kernel=32, u_max=2e4
            )

            current_err = max_rel(current, ref_kernel)
            stable_err = max_rel(stable32, ref_kernel)
            proj_current_err = max_rel(proj_current, ref_proj)
            proj_stable_err = max_rel(proj_stable, ref_proj)
            finite_current = bool(np.isfinite(np.asarray(current)).all())
            print(
                f"{case.name:<28} {current_err:>13.3e} {stable_err:>13.3e} "
                f"{proj_current_err:>11.3e} {proj_stable_err:>14.3e} "
                f"{str(finite_current):>10}"
            )


if __name__ == "__main__":
    main()
