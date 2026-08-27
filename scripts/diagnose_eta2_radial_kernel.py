from __future__ import annotations

import os

os.environ.setdefault("JEANSPY_JAX_PLATFORM", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> None:
    import numpy as np
    import jax
    import jax.numpy as jnp

    import jeanspy.model_numpyro as mn
    from jeanspy.baes_eta2 import BaesEta2AnisotropyModel
    from jeanspy.model_numpyro import ConstantAnisotropyModel, DSphModel, NFWModel, PlummerModel

    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("This diagnostic requires JAX float64.")

    dtype = jnp.float64
    re_pc = 220.0
    beta_values = (0.7, 0.9, 0.98, 0.995)
    u_probe_np = np.asarray([1.000001, 1.01, 1.1, 2.0, 10.0, 100.0, 1e3, 1e4, 1e5])
    u_probe = jnp.asarray(u_probe_np, dtype=dtype)
    R_probe = jnp.full_like(u_probe, re_pc)
    n_kernel_values = (16, 32, 64, 128, 256, 512)

    R_over_re = np.asarray([0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0])
    R = jnp.asarray(R_over_re * re_pc, dtype=dtype)
    projection_u_max = 2e4
    projection_n_u = 512

    def max_rel(a, b) -> float:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        scale = max(float(np.max(np.abs(bb))), 1e-12)
        floor = max(scale * 1e-12, 1e-14)
        return float(np.max(np.abs(aa - bb) / np.maximum(np.abs(bb), floor)))

    def common_params() -> dict[str, jnp.ndarray]:
        raw = {
            "re_pc": re_pc,
            "rs_pc": 5.0 * re_pc,
            "rhos_Msunpc3": 7.5e-3,
            "r_t_pc": 40.0 * re_pc,
            "vmem_kms": 0.0,
        }
        return {k: jnp.asarray(v, dtype=dtype) for k, v in raw.items()}

    def make_dsph(ani):
        return DSphModel(
            submodels={
                "StellarModel": PlummerModel(),
                "DMModel": NFWModel(),
                "AnisotropyModel": ani,
            }
        )

    def eta2_projection(dsph, params, *, n_kernel: int):
        stellar = dsph.submodels["StellarModel"]
        dm = dsph.submodels["DMModel"]
        ani = dsph.submodels["AnisotropyModel"]

        @jax.jit
        def _eval():
            R2d = R[:, None]
            s_max = jnp.sqrt(jnp.log(jnp.asarray(projection_u_max, dtype=dtype)))
            s = jnp.linspace(jnp.asarray(0.0, dtype=dtype), s_max, projection_n_u)
            u = jnp.exp(s * s)
            u2d = u[None, :]
            r = R2d * u2d
            nu3 = stellar.density_3d(r, re_pc=params["re_pc"])
            sigma2 = stellar.density_2d(R2d, re_pc=params["re_pc"])
            mass = dm.enclosed_mass(r, method="analytic", params=params)
            grav = (mn.GMsun_m3s2 * mass / mn.PARSEC_M) * 1e-6
            K = ani.kernel(u2d, R2d, params=params, n_kernel=n_kernel)
            integrand = 2.0 * K * (nu3 / sigma2) * grav * (2.0 * s)[None, :]
            return mn._simpson_uniform_last_axis(integrand, s[1] - s[0])

        return jax.block_until_ready(_eval())

    print("eta=2 radial-kernel convergence diagnostic (CPU float64)")
    print(f"u probes: {u_probe_np.tolist()}")

    for beta in beta_values:
        base = common_params()
        eta_params = dict(base)
        eta_params.update(
            {
                "beta_0": jnp.asarray(beta, dtype=dtype),
                "beta_inf": jnp.asarray(beta, dtype=dtype),
                "r_a": jnp.asarray(re_pc, dtype=dtype),
            }
        )
        const_params = dict(base)
        const_params["beta_ani"] = jnp.asarray(beta, dtype=dtype)

        eta = BaesEta2AnisotropyModel()
        const = ConstantAnisotropyModel()
        eta_dsph = make_dsph(eta)
        const_dsph = make_dsph(const)

        # SciPy/hypergeometric path is an independent constant-beta reference.
        k_ref = jax.block_until_ready(
            const.kernel(u_probe, R_probe, params=const_params, backend="scipy")
        )
        k_const_jax = jax.block_until_ready(
            const.kernel(u_probe, R_probe, params=const_params, backend="jax", n_kernel=256)
        )
        print(f"\nbeta={beta:+.3f}")
        print(f"constant JAX(256) vs SciPy kernel max rel: {max_rel(k_const_jax, k_ref):.3e}")
        print("n_kernel   kernel max rel vs SciPy   projection max rel vs constant")

        const_projection = jax.block_until_ready(
            const_dsph.sigmalos2(
                R,
                params=const_params,
                backend="kernel",
                n_u=4096,
                n_kernel=256,
                u_max=projection_u_max,
                constant_kernel_backend="jax",
                dm_mass_method="analytic",
                jit=True,
            )
        )

        for n_kernel in n_kernel_values:
            k_eta = jax.block_until_ready(
                eta.kernel(u_probe, R_probe, params=eta_params, n_kernel=n_kernel)
            )
            k_err = max_rel(k_eta, k_ref)
            eta_proj = eta2_projection(eta_dsph, eta_params, n_kernel=n_kernel)
            proj_err = max_rel(eta_proj, const_projection)
            print(f"{n_kernel:>8d}   {k_err:>23.3e}   {proj_err:>27.3e}")

        # Show where the n_kernel=64 error grows with u.
        k_eta64 = np.asarray(
            jax.block_until_ready(eta.kernel(u_probe, R_probe, params=eta_params, n_kernel=64)),
            dtype=np.float64,
        )
        k_ref_np = np.asarray(k_ref, dtype=np.float64)
        rel = np.abs(k_eta64 - k_ref_np) / np.maximum(np.abs(k_ref_np), 1e-14)
        print("n_kernel=64 relative error by u:")
        print("  " + ", ".join(f"u={u:g}:{e:.2e}" for u, e in zip(u_probe_np, rel)))


if __name__ == "__main__":
    main()
