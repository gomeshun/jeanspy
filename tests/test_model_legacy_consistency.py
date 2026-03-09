import os
import unittest
import warnings
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jeanspy.model import (
    BaesAnisotropyModel as BaesLegacy,
    ConstantAnisotropyModel as ConstantLegacy,
    DSphModel as DSphLegacy,
    NFWModel as NFWLegacy,
    OsipkovMerrittModel as OsipkovMerrittLegacy,
    PlummerModel as PlummerLegacy,
    ZhaoModel as ZhaoLegacy,
)
from jeanspy.model_numpyro import (
    GMsun_m3s2,
    PARSEC_M,
    BaesAnisotropyModel as BaesNumPyro,
    ConstantAnisotropyModel as ConstantNumPyro,
    DSphModel as DSphNumPyro,
    NFWModel as NFWNumPyro,
    OsipkovMerrittModel as OsipkovMerrittNumPyro,
    PlummerModel as PlummerNumPyro,
    ZhaoModel as ZhaoNumPyro,
)


def _assert_all_finite(testcase: unittest.TestCase, values: Any, *, label: str) -> None:
    arr = np.asarray(values, dtype=np.float64)
    testcase.assertTrue(np.isfinite(arr).all(), msg=f"{label} contains non-finite values: {arr}")


def _assert_allclose(
    testcase: unittest.TestCase,
    legacy: Any,
    numpyro: Any,
    *,
    label: str,
    rtol: float,
    atol: float = 1e-12,
) -> None:
    legacy_arr = np.asarray(legacy, dtype=np.float64)
    numpyro_arr = np.asarray(numpyro, dtype=np.float64)
    _assert_all_finite(testcase, legacy_arr, label=f"{label} (legacy)")
    _assert_all_finite(testcase, numpyro_arr, label=f"{label} (numpyro)")
    np.testing.assert_allclose(legacy_arr, numpyro_arr, rtol=rtol, atol=atol, err_msg=label)


def _legacy_dm_from_params(params: dict[str, float]):
    if {"a", "b", "g"} <= params.keys():
        return ZhaoLegacy(
            rs_pc=params["rs_pc"],
            rhos_Msunpc3=params["rhos_Msunpc3"],
            a=params["a"],
            b=params["b"],
            g=params["g"],
            r_t_pc=params["r_t_pc"],
        )
    return NFWLegacy(
        rs_pc=params["rs_pc"],
        rhos_Msunpc3=params["rhos_Msunpc3"],
        r_t_pc=params["r_t_pc"],
    )


def _numpyro_dm_from_params(params: dict[str, float]):
    if {"a", "b", "g"} <= params.keys():
        return ZhaoNumPyro()
    return NFWNumPyro()


def _legacy_anisotropy_from_kind(kind: str, params: dict[str, float]):
    if kind == "constant":
        return ConstantLegacy(beta_ani=params["beta_ani"])
    if kind == "osipkov_merritt":
        return OsipkovMerrittLegacy(r_a=params["r_a"])
    if kind == "baes":
        return BaesLegacy(
            beta_0=params["beta_0"],
            beta_inf=params["beta_inf"],
            r_a=params["r_a"],
            eta=params["eta"],
        )
    raise ValueError(f"Unknown anisotropy kind: {kind}")


def _numpyro_anisotropy_from_kind(kind: str):
    if kind == "constant":
        return ConstantNumPyro()
    if kind == "osipkov_merritt":
        return OsipkovMerrittNumPyro()
    if kind == "baes":
        return BaesNumPyro()
    raise ValueError(f"Unknown anisotropy kind: {kind}")


def _numpyro_sigmalos2_integrand(params: dict[str, float], anisotropy_kind: str, u: np.ndarray, R_pc: np.ndarray) -> np.ndarray:
    stellar = PlummerNumPyro()
    dm = _numpyro_dm_from_params(params)
    anisotropy = _numpyro_anisotropy_from_kind(anisotropy_kind)

    R_2d = np.asarray(R_pc, dtype=np.float64)[:, None]
    u_2d = np.asarray(u, dtype=np.float64)[None, :]
    r_2d = R_2d * u_2d

    nu_3d = np.asarray(stellar.density_3d(jnp.asarray(r_2d), re_pc=params["re_pc"]), dtype=np.float64)
    sigma_2d = np.asarray(stellar.density_2d(jnp.asarray(R_2d), re_pc=params["re_pc"]), dtype=np.float64)
    mass = np.asarray(dm.enclosed_mass(jnp.asarray(r_2d), params=params, method="analytic"), dtype=np.float64)
    kernel = np.asarray(anisotropy.kernel(jnp.asarray(u_2d), jnp.asarray(R_2d), params=params), dtype=np.float64)

    return 2.0 * (kernel / u_2d) * (nu_3d / sigma_2d) * (GMsun_m3s2 * mass / PARSEC_M) * 1e-6


class TestSharedModelMethodsConsistency(unittest.TestCase):
    def test_plummer_density_matches_legacy(self):
        radii = np.geomspace(1e-3, 1e4, 64)
        cases = [
            {"re_pc": 3.0},
            {"re_pc": 200.0},
            {"re_pc": 5e3},
        ]

        model_numpyro = PlummerNumPyro()
        for params in cases:
            with self.subTest(params=params):
                model_legacy = PlummerLegacy(re_pc=params["re_pc"])
                _assert_allclose(
                    self,
                    model_legacy.density_2d(radii),
                    model_numpyro.density_2d(jnp.asarray(radii), re_pc=params["re_pc"]),
                    label=f"Plummer.density_2d {params}",
                    rtol=5e-7,
                )
                _assert_allclose(
                    self,
                    model_legacy.density_3d(radii),
                    model_numpyro.density_3d(jnp.asarray(radii), re_pc=params["re_pc"]),
                    label=f"Plummer.density_3d {params}",
                    rtol=5e-7,
                )

    def test_nfw_density_and_enclosed_mass_match_legacy(self):
        cases = [
            {
                "params": {"rs_pc": 0.3, "rhos_Msunpc3": 1e-6, "r_t_pc": 1.0},
                "mass_rtol": 1e-9,
            },
            {
                "params": {"rs_pc": 1200.0, "rhos_Msunpc3": 1e-2, "r_t_pc": 8000.0},
                "mass_rtol": 2e-4,
            },
            {
                "params": {"rs_pc": 5e4, "rhos_Msunpc3": 100.0, "r_t_pc": 1e6},
                "mass_rtol": 6e-3,
            },
        ]

        radii = np.geomspace(1e-3, 1e5, 96)
        model_numpyro = NFWNumPyro()
        for case in cases:
            params = case["params"]
            with self.subTest(params=params):
                model_legacy = NFWLegacy(**params)
                _assert_allclose(
                    self,
                    model_legacy.mass_density_3d(radii),
                    model_numpyro.mass_density_3d(jnp.asarray(radii), params=params),
                    label=f"NFW.mass_density_3d {params}",
                    rtol=1e-12,
                )
                _assert_allclose(
                    self,
                    model_legacy.enclosure_mass(radii),
                    model_numpyro.enclosed_mass(jnp.asarray(radii), params=params, method="analytic"),
                    label=f"NFW.enclosed_mass {params}",
                    rtol=case["mass_rtol"],
                    atol=1e-10,
                )

    def test_zhao_density_and_enclosed_mass_match_legacy(self):
        cases = [
            {"rs_pc": 420.0, "rhos_Msunpc3": 0.06, "a": 1.2, "b": 4.0, "g": 0.6, "r_t_pc": 5000.0},
            {"rs_pc": 50.0, "rhos_Msunpc3": 1e-4, "a": 0.5, "b": 6.0, "g": 0.1, "r_t_pc": 1e4},
            {"rs_pc": 2000.0, "rhos_Msunpc3": 5.0, "a": 3.0, "b": 8.0, "g": 1.8, "r_t_pc": 2e4},
        ]

        radii = np.geomspace(1e-3, 1e5, 96)
        model_numpyro = ZhaoNumPyro()
        for params in cases:
            with self.subTest(params=params):
                model_legacy = ZhaoLegacy(**params)
                _assert_allclose(
                    self,
                    model_legacy.mass_density_3d(radii),
                    model_numpyro.mass_density_3d(jnp.asarray(radii), params=params),
                    label=f"Zhao.mass_density_3d {params}",
                    rtol=1e-12,
                )
                _assert_allclose(
                    self,
                    model_legacy.enclosure_mass(radii),
                    model_numpyro.enclosed_mass(jnp.asarray(radii), params=params, method="analytic"),
                    label=f"Zhao.enclosed_mass {params}",
                    rtol=2e-10,
                    atol=1e-10,
                )

    def test_constant_anisotropy_methods_match_legacy(self):
        radii = np.geomspace(0.1, 10.0, 64)
        u = np.geomspace(1.0 + 1e-6, 1e3, 256)
        R_pc = np.array([0.3, 3.0, 300.0], dtype=np.float64)[:, None]

        cases = [
            {"beta_ani": -10.0, "kernel_rtol": 1e-10},
            {"beta_ani": -2.0, "kernel_rtol": 1e-12},
            {"beta_ani": 0.2, "kernel_rtol": 2e-2},
            {"beta_ani": 0.95, "kernel_rtol": 1e-7},
        ]

        model_numpyro = ConstantNumPyro()
        for case in cases:
            params = {"beta_ani": case["beta_ani"]}
            with self.subTest(params=params):
                model_legacy = ConstantLegacy(**params)
                _assert_allclose(
                    self,
                    model_legacy.beta(radii),
                    model_numpyro.beta(jnp.asarray(radii), params=params),
                    label=f"Constant.beta {params}",
                    rtol=0.0,
                )
                _assert_allclose(
                    self,
                    model_legacy.f(radii),
                    model_numpyro.f(jnp.asarray(radii), params=params),
                    label=f"Constant.f {params}",
                    rtol=0.0,
                )
                _assert_allclose(
                    self,
                    model_legacy.kernel(u, R_pc),
                    model_numpyro.kernel(jnp.asarray(u), jnp.asarray(R_pc), params=params),
                    label=f"Constant.kernel {params}",
                    rtol=case["kernel_rtol"],
                    atol=1e-10,
                )

    def test_osipkov_merritt_methods_match_legacy(self):
        radii = np.geomspace(0.1, 1e4, 64)
        u = np.geomspace(1.0 + 1e-6, 1e3, 256)
        R_pc = np.array([0.3, 3.0, 300.0], dtype=np.float64)[:, None]

        model_numpyro = OsipkovMerrittNumPyro()
        for r_a in (0.1, 1.0, 350.0, 1e4):
            params = {"r_a": r_a}
            with self.subTest(params=params):
                model_legacy = OsipkovMerrittLegacy(**params)
                _assert_allclose(
                    self,
                    model_legacy.beta(radii),
                    model_numpyro.beta(jnp.asarray(radii), params=params),
                    label=f"OsipkovMerritt.beta {params}",
                    rtol=0.0,
                )
                _assert_allclose(
                    self,
                    model_legacy.f(radii),
                    model_numpyro.f(jnp.asarray(radii), params=params),
                    label=f"OsipkovMerritt.f {params}",
                    rtol=0.0,
                )
                _assert_allclose(
                    self,
                    model_legacy.kernel(u[None, :], R_pc),
                    model_numpyro.kernel(jnp.asarray(u)[None, :], jnp.asarray(R_pc), params=params),
                    label=f"OsipkovMerritt.kernel {params}",
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_baes_methods_match_legacy(self):
        u = np.geomspace(1.0 + 1e-6, 1e3, 256)
        R_pc = np.array([0.5, 10.0], dtype=np.float64)

        cases = [
            {"beta_0": -10.0, "beta_inf": -10.0, "r_a": 1.0, "eta": 2.0},
            {"beta_0": -10.0, "beta_inf": 0.8, "r_a": 1.0, "eta": 2.0},
            {"beta_0": 0.0, "beta_inf": 1.0, "r_a": 300.0, "eta": 2.0},
            {"beta_0": 0.8, "beta_inf": -0.5, "r_a": 100.0, "eta": 6.0},
        ]

        model_numpyro = BaesNumPyro()
        for params in cases:
            radii = np.geomspace(max(0.1, params["r_a"] * 0.1), params["r_a"] * 10.0, 64)
            with self.subTest(params=params):
                model_legacy = BaesLegacy(**params)
                _assert_allclose(
                    self,
                    model_legacy.beta(radii),
                    model_numpyro.beta(jnp.asarray(radii), params=params),
                    label=f"Baes.beta {params}",
                    rtol=0.0,
                )
                _assert_allclose(
                    self,
                    model_legacy.f(radii),
                    model_numpyro.f(jnp.asarray(radii), params=params),
                    label=f"Baes.f {params}",
                    rtol=0.0,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    legacy_kernel = model_legacy.kernel(u, R_pc, n=320)
                numpyro_kernel = model_numpyro.kernel(
                    jnp.asarray(u)[None, :],
                    jnp.asarray(R_pc)[:, None],
                    params=params,
                    n_kernel=320,
                )
                _assert_allclose(
                    self,
                    legacy_kernel,
                    numpyro_kernel,
                    label=f"Baes.kernel {params}",
                    rtol=2e-5,
                    atol=1e-8,
                )

    def test_zhao_enclosed_mass_fails_at_nfw_limit_in_legacy_impl(self):
        params = {
            "rs_pc": 1200.0,
            "rhos_Msunpc3": 1e-2,
            "a": 1.0,
            "b": 3.0,
            "g": 1.0,
            "r_t_pc": 8000.0,
        }
        radii = np.geomspace(1.0, 1e4, 32)

        legacy = ZhaoLegacy(**params).enclosure_mass(radii)
        numpyro = ZhaoNumPyro().enclosed_mass(jnp.asarray(radii), params=params, method="analytic")

        _assert_all_finite(self, numpyro, label="Zhao NFW-limit enclosed_mass (numpyro)")
        _assert_all_finite(self, legacy, label="Zhao NFW-limit enclosed_mass (legacy)")
        _assert_allclose(
            self,
            legacy,
            numpyro,
            label="Zhao NFW-limit enclosed_mass",
            rtol=1e-10,
            atol=1e-10,
        )


class TestDSphConsistencyAgainstLegacy(unittest.TestCase):
    def _make_legacy_dsph(self, params: dict[str, float], anisotropy_kind: str) -> DSphLegacy:
        return DSphLegacy(
            vmem_kms=params["vmem_kms"],
            submodels={
                "StellarModel": PlummerLegacy(re_pc=params["re_pc"]),
                "DMModel": _legacy_dm_from_params(params),
                "AnisotropyModel": _legacy_anisotropy_from_kind(anisotropy_kind, params),
            },
        )

    def _make_numpyro_dsph(self, params: dict[str, float], anisotropy_kind: str) -> DSphNumPyro:
        return DSphNumPyro(
            submodels={
                "StellarModel": PlummerNumPyro(),
                "DMModel": _numpyro_dm_from_params(params),
                "AnisotropyModel": _numpyro_anisotropy_from_kind(anisotropy_kind),
            }
        )

    def test_sigmalos2_integrand_matches_legacy(self):
        cases = [
            {
                "name": "constant_nfw_typical",
                "anisotropy_kind": "constant",
                "params": {
                    "re_pc": 200.0,
                    "rs_pc": 1200.0,
                    "rhos_Msunpc3": 1e-2,
                    "r_t_pc": 8000.0,
                    "beta_ani": 0.2,
                    "vmem_kms": 0.0,
                },
                "rtol": 2e-2,
            },
            {
                "name": "constant_zhao_boundary",
                "anisotropy_kind": "constant",
                "params": {
                    "re_pc": 50.0,
                    "rs_pc": 300.0,
                    "rhos_Msunpc3": 0.2,
                    "a": 0.7,
                    "b": 5.5,
                    "g": 1.2,
                    "r_t_pc": 2e4,
                    "beta_ani": 0.95,
                    "vmem_kms": 0.0,
                },
                "rtol": 1e-6,
            },
            {
                "name": "osipkov_nfw_extreme_ra",
                "anisotropy_kind": "osipkov_merritt",
                "params": {
                    "re_pc": 220.0,
                    "rs_pc": 1100.0,
                    "rhos_Msunpc3": 7.5e-3,
                    "r_t_pc": 9000.0,
                    "r_a": 0.1,
                    "vmem_kms": 0.0,
                },
                "rtol": 1e-9,
            },
        ]

        u = np.geomspace(1.0 + 1e-6, 100.0, 120)
        for case in cases:
            params = case["params"]
            R_pc = np.geomspace(max(0.1, params["re_pc"] * 0.05), params["re_pc"] * 2.0, 4)
            with self.subTest(case=case["name"]):
                legacy_dsph = self._make_legacy_dsph(params, case["anisotropy_kind"])
                legacy_integrand = legacy_dsph.integrand_sigmalos2(u, R_pc, n_kernel=256)
                numpyro_integrand = _numpyro_sigmalos2_integrand(params, case["anisotropy_kind"], u, R_pc)
                _assert_allclose(
                    self,
                    legacy_integrand,
                    numpyro_integrand,
                    label=f"DSph.integrand_sigmalos2 {case['name']}",
                    rtol=case["rtol"],
                    atol=1e-10,
                )

    def test_sigmalos2_matches_legacy_for_constant_and_osipkov_merritt(self):
        cases = [
            {
                "name": "constant_nfw_typical",
                "anisotropy_kind": "constant",
                "params": {
                    "re_pc": 200.0,
                    "rs_pc": 1200.0,
                    "rhos_Msunpc3": 1e-2,
                    "r_t_pc": 8000.0,
                    "beta_ani": 0.2,
                    "vmem_kms": 0.0,
                },
                "rtol": 4e-3,
            },
            {
                "name": "constant_zhao_boundary",
                "anisotropy_kind": "constant",
                "params": {
                    "re_pc": 50.0,
                    "rs_pc": 300.0,
                    "rhos_Msunpc3": 0.2,
                    "a": 0.7,
                    "b": 5.5,
                    "g": 1.2,
                    "r_t_pc": 2e4,
                    "beta_ani": 0.95,
                    "vmem_kms": 0.0,
                },
                "rtol": 5e-4,
            },
            {
                "name": "osipkov_nfw_extreme_ra",
                "anisotropy_kind": "osipkov_merritt",
                "params": {
                    "re_pc": 220.0,
                    "rs_pc": 1100.0,
                    "rhos_Msunpc3": 7.5e-3,
                    "r_t_pc": 9000.0,
                    "r_a": 0.1,
                    "vmem_kms": 0.0,
                },
                "rtol": 1e-5,
            },
        ]

        for case in cases:
            params = case["params"]
            R_pc = np.geomspace(max(0.1, params["re_pc"] * 0.05), params["re_pc"] * 5.0, 10)
            with self.subTest(case=case["name"]):
                legacy_dsph = self._make_legacy_dsph(params, case["anisotropy_kind"])
                numpyro_dsph = self._make_numpyro_dsph(params, case["anisotropy_kind"])

                legacy_sigmalos2 = legacy_dsph.sigmalos2_dequad(R_pc, n=2048, n_kernel=256)
                numpyro_sigmalos2 = numpyro_dsph.sigmalos2(
                    jnp.asarray(R_pc),
                    params=params,
                    backend="kernel",
                    n_u=1024,
                    u_max=5000.0,
                )
                _assert_allclose(
                    self,
                    legacy_sigmalos2,
                    numpyro_sigmalos2,
                    label=f"DSph.sigmalos2 {case['name']}",
                    rtol=case["rtol"],
                    atol=1e-8,
                )

    def test_sigmalos2_with_baes_is_consistent(self):
        params = {
            "re_pc": 220.0,
            "rs_pc": 1100.0,
            "rhos_Msunpc3": 7.5e-3,
            "r_t_pc": 9000.0,
            "beta_0": 0.0,
            "beta_inf": 0.65,
            "r_a": 300.0,
            "eta": 2.2,
            "vmem_kms": 0.0,
        }
        R_pc = np.geomspace(20.0, 1200.0, 8)

        legacy_dsph = self._make_legacy_dsph(params, "baes")
        numpyro_dsph = self._make_numpyro_dsph(params, "baes")

        legacy_sigmalos2 = legacy_dsph.sigmalos2_dequad(R_pc, n=1024, n_kernel=128)
        numpyro_sigmalos2 = numpyro_dsph.sigmalos2(
            jnp.asarray(R_pc),
            params=params,
            backend="kernel",
            n_u=768,
            u_max=3000.0,
        )
        _assert_allclose(
            self,
            legacy_sigmalos2,
            numpyro_sigmalos2,
            label="DSph.sigmalos2 baes",
            rtol=1e-2,
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
