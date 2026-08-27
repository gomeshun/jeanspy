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
    from jeanspy.model_numpyro import (
        ConstantAnisotropyModel,
        DSphModel,
        NFWModel,
        PlummerModel,
    )

    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("This diagnostic requires JAX float64.")

    dtype = jnp.float64
    re_pc = 220.0
    R_over_re = np.asarray([0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0])
    R = jnp.asarray(R_over_re * re_pc, dtype=dtype)
    beta_values = (-5.0, 0.7, 0.98)
    u_max_values = (5e3, 2e4, 1e5, 5e5)
    n_u = 512
    n_kernel = 64
    abel_u_max = 5e5
    abel_n_r = 65536

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

    def sqrtlog_projection(dsph, params, *, u_max: float):
        stellar = dsph.submodels["StellarModel"]
        dm = dsph.submodels["DMModel"]
        ani = dsph.submodels["AnisotropyModel"]

        @jax.jit
        def _eval():
            R2d = R[:, None]
            s_max = jnp.sqrt(jnp.log(jnp.asarray(u_max, dtype=dtype)))
            s = jnp.linspace(jnp.asarray(0.0, dtype=dtype), s_max, n_u)
            t = s * s
            u = jnp.exp(t)
            u2d = u[None, :]
            r = R2d * u2d

            nu3 = stellar.density_3d(r, re_pc=params["re_pc"])
            sigma2 = stellar.density_2d(R2d, re_pc=params["re_pc"])
            mass = dm.enclosed_mass(r, method="analytic", params=params)
            grav = (mn.GMsun_m3s2 * mass / mn.PARSEC_M) * 1e-6

            if isinstance(ani, ConstantAnisotropyModel):
                K = ani.kernel(
                    u2d,
                    R2d,
                    params=params,
                    backend="jax",
                    n_kernel=n_kernel,
                )
            else:
                K = ani.kernel(u2d, R2d, params=params, n_kernel=n_kernel)

            integrand_s = 2.0 * K * (nu3 / sigma2) * grav * (2.0 * s)[None, :]
            h = s[1] - s[0]
            return mn._simpson_uniform_last_axis(integrand_s, h)

        return jax.block_until_ready(_eval())

    def max_rel(a, b) -> float:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        scale = max(float(np.max(np.abs(bb))), 1e-12)
        floor = max(scale * 1e-10, 1e-12)
        return float(np.max(np.abs(aa - bb) / np.maximum(np.abs(bb), floor)))

    print("eta=2 radial-tail diagnostic (CPU float64)")
    print(
        f"R/Re={R_over_re.tolist()}, n_u={n_u}, n_kernel={n_kernel}, "
        f"Abel reference n_r={abel_n_r}, u_max={abel_u_max:g}"
    )

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

        eta_dsph = make_dsph(BaesEta2AnisotropyModel())
        const_dsph = make_dsph(ConstantAnisotropyModel())

        # Exact constant-anisotropy limit should make both kernel models agree.
        u_probe = jnp.asarray(np.geomspace(1.0 + 1e-8, 5e5, 256), dtype=dtype)
        R_probe = jnp.full_like(u_probe, re_pc)
        K_eta = jax.block_until_ready(
            eta_dsph.submodels["AnisotropyModel"].kernel(
                u_probe, R_probe, params=eta_params, n_kernel=n_kernel
            )
        )
        K_const = jax.block_until_ready(
            const_dsph.submodels["AnisotropyModel"].kernel(
                u_probe,
                R_probe,
                params=const_params,
                backend="jax",
                n_kernel=n_kernel,
            )
        )
        kernel_limit_err = max_rel(K_eta, K_const)

        abel = jax.block_until_ready(
            const_dsph.sigmalos2(
                R,
                params=const_params,
                backend="abel",
                n_r=abel_n_r,
                u_max=abel_u_max,
                r_min_factor=0.35,
                dm_mass_method="analytic",
                jit=True,
            )
        )

        print(f"\nbeta={beta:+.2f}: eta2-vs-constant kernel max rel={kernel_limit_err:.3e}")
        print("u_max      eta2-vs-constant    eta2-vs-Abel    const-vs-Abel    worst R/Re")
        for u_max in u_max_values:
            eta_val = sqrtlog_projection(eta_dsph, eta_params, u_max=u_max)
            const_val = sqrtlog_projection(const_dsph, const_params, u_max=u_max)
            eta_const = max_rel(eta_val, const_val)
            eta_abel = max_rel(eta_val, abel)
            const_abel = max_rel(const_val, abel)

            eta_np = np.asarray(eta_val, dtype=np.float64)
            abel_np = np.asarray(abel, dtype=np.float64)
            rel_by_r = np.abs(eta_np - abel_np) / np.maximum(np.abs(abel_np), 1e-12)
            worst_r = float(R_over_re[int(np.argmax(rel_by_r))])
            print(
                f"{u_max:>8.0f}   {eta_const:>17.3e}   {eta_abel:>13.3e}   "
                f"{const_abel:>13.3e}   {worst_r:>10.3g}"
            )


if __name__ == "__main__":
    main()
