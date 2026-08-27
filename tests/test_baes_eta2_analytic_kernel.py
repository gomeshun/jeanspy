import numpy as np
from scipy.integrate import quad

import jax.numpy as jnp

from jeanspy.analytic_kernels import baes_eta2_kernel_appell
from jeanspy.model_numpyro import (
    BaesAnisotropyModel,
    ConstantAnisotropyModel,
    DSphModel,
    NFWModel,
    OsipkovMerrittModel,
    PlummerModel,
)


def _baes_eta2_kernel_direct(u, R, beta_0, beta_inf, r_a):
    """Independent scalar reference from the original kernel definition."""

    def beta(r):
        x = (r / r_a) ** 2
        return (beta_0 + beta_inf * x) / (1.0 + x)

    def f(r):
        x = (r / r_a) ** 2
        return r ** (2.0 * beta_0) * (1.0 + x) ** (beta_inf - beta_0)

    if u == 1.0:
        return 0.0

    t_max = np.sqrt(u * u - 1.0)

    def integrand(t):
        u_int = np.sqrt(1.0 + t * t)
        r_int = R * u_int
        return (1.0 - beta(r_int) / (u_int * u_int)) / f(r_int)

    inner, _ = quad(integrand, 0.0, t_max, epsabs=1.0e-12, epsrel=1.0e-12)
    return f(R * u) / u * inner


def test_baes_eta2_appell_matches_original_kernel_integral():
    cases = [
        (1.15, 80.0, -0.5, 0.6, 250.0),
        (2.5, 150.0, 0.0, 0.8, 300.0),
        (8.0, 60.0, -1.5, 0.2, 120.0),
        (25.0, 300.0, 0.25, 0.75, 100.0),
    ]

    for u, R, beta_0, beta_inf, r_a in cases:
        analytic = float(
            baes_eta2_kernel_appell(
                u,
                R,
                beta_0=beta_0,
                beta_inf=beta_inf,
                r_a=r_a,
            )
        )
        direct = _baes_eta2_kernel_direct(u, R, beta_0, beta_inf, r_a)
        np.testing.assert_allclose(analytic, direct, rtol=2.0e-10, atol=2.0e-12)


def test_baes_eta2_appell_matches_existing_jax_kernel():
    u = np.geomspace(1.0 + 5.0e-4, 30.0, 48)[None, :]
    R = np.asarray([50.0, 180.0])[:, None]
    ani = BaesAnisotropyModel()

    parameter_sets = [
        {"beta_0": -0.5, "beta_inf": 0.6, "r_a": 250.0, "eta": 2.0},
        {"beta_0": 0.0, "beta_inf": 0.8, "r_a": 300.0, "eta": 2.0},
        {"beta_0": -2.0, "beta_inf": 0.2, "r_a": 120.0, "eta": 2.0},
    ]

    for params in parameter_sets:
        analytic = baes_eta2_kernel_appell(
            u,
            R,
            beta_0=params["beta_0"],
            beta_inf=params["beta_inf"],
            r_a=params["r_a"],
        )
        numerical = np.asarray(
            ani.kernel(
                jnp.asarray(u),
                jnp.asarray(R),
                params=params,
                n_kernel=512,
            )
        )

        assert np.isfinite(analytic).all()
        assert np.isfinite(numerical).all()
        np.testing.assert_allclose(analytic, numerical, rtol=8.0e-3, atol=3.0e-5)


def test_baes_eta2_appell_reproduces_constant_and_osipkov_limits():
    u = np.geomspace(1.0 + 1.0e-4, 20.0, 80)[None, :]
    R = np.asarray([60.0, 200.0])[:, None]

    beta_const = 0.3
    analytic_const = baes_eta2_kernel_appell(
        u,
        R,
        beta_0=beta_const,
        beta_inf=beta_const,
        r_a=170.0,
    )
    const_model = ConstantAnisotropyModel()
    constant = np.asarray(
        const_model.kernel(
            jnp.asarray(u),
            jnp.asarray(R),
            params={"beta_ani": beta_const},
            backend="scipy",
        )
    )
    constant = np.broadcast_to(constant, analytic_const.shape)
    np.testing.assert_allclose(analytic_const, constant, rtol=2.0e-5, atol=2.0e-7)

    r_a = 170.0
    analytic_om = baes_eta2_kernel_appell(
        u,
        R,
        beta_0=0.0,
        beta_inf=1.0,
        r_a=r_a,
    )
    om_model = OsipkovMerrittModel()
    om = np.asarray(
        om_model.kernel(
            jnp.asarray(u),
            jnp.asarray(R),
            params={"r_a": r_a},
        )
    )
    np.testing.assert_allclose(analytic_om, om, rtol=2.0e-5, atol=2.0e-7)


class _BaesEta2AppellForTest(BaesAnisotropyModel):
    """Test-only adapter that injects the analytic kernel into DSphModel."""

    def kernel(self, u, R_pc, *, params, n_kernel=128):
        del n_kernel
        value = baes_eta2_kernel_appell(
            np.asarray(u),
            np.asarray(R_pc),
            beta_0=float(np.asarray(params["beta_0"])),
            beta_inf=float(np.asarray(params["beta_inf"])),
            r_a=float(np.asarray(params["r_a"])),
        )
        return jnp.asarray(value, dtype=jnp.result_type(u, R_pc))


def test_baes_eta2_appell_sigmalos_matches_kernel_free_abel_solver():
    model = DSphModel(
        submodels={
            "StellarModel": PlummerModel(),
            "DMModel": NFWModel(),
            "AnisotropyModel": _BaesEta2AppellForTest(),
        }
    )
    params = {
        "re_pc": 220.0,
        "rs_pc": 1100.0,
        "rhos_Msunpc3": 7.5e-3,
        "r_t_pc": 9000.0,
        "beta_0": 0.0,
        "beta_inf": 0.65,
        "r_a": 300.0,
        "eta": 2.0,
        "vmem_kms": 0.0,
    }
    R = jnp.asarray([50.0, 120.0, 300.0])

    sigma2_kernel = np.asarray(
        model.sigmalos2(
            R,
            params=params,
            backend="kernel",
            jit=False,
            n_u=512,
            n_kernel=128,
            u_max=2000.0,
            use_analytic_dm=True,
        )
    )
    sigma2_abel = np.asarray(
        model.sigmalos2(
            R,
            params=params,
            backend="abel",
            jit=False,
            n_r=1024,
            u_max=2000.0,
            use_analytic_dm=True,
        )
    )

    assert np.isfinite(sigma2_kernel).all()
    assert np.isfinite(sigma2_abel).all()
    assert np.all(sigma2_kernel > 0.0)
    assert np.all(sigma2_abel > 0.0)
    np.testing.assert_allclose(sigma2_kernel, sigma2_abel, rtol=6.0e-2, atol=2.0e-3)
