import os
import importlib
import unittest
import warnings
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import jeanspy.model_numpyro as _model_numpyro_mod

model_numpyro_mod = importlib.reload(_model_numpyro_mod)

from jeanspy.model import (
    BaesAnisotropyModel as BaesClassical,
    ConstantAnisotropyModel as ConstantClassical,
    DSphModel as DSphClassical,
    NFWModel as NFWClassical,
    OsipkovMerrittModel as OsipkovMerrittClassical,
    PlummerModel as PlummerClassical,
    ZhaoModel as ZhaoClassical,
)

GMsun_m3s2 = model_numpyro_mod.GMsun_m3s2
PARSEC_M = model_numpyro_mod.PARSEC_M
BaesNumPyro = model_numpyro_mod.BaesAnisotropyModel
ConstantNumPyro = model_numpyro_mod.ConstantAnisotropyModel
DSphNumPyro = model_numpyro_mod.DSphModel
NFWNumPyro = model_numpyro_mod.NFWModel
OsipkovMerrittNumPyro = model_numpyro_mod.OsipkovMerrittModel
PlummerNumPyro = model_numpyro_mod.PlummerModel
ZhaoNumPyro = model_numpyro_mod.ZhaoModel


def _assert_all_finite(testcase: unittest.TestCase, values: Any, *, label: str) -> None:
    arr = np.asarray(values, dtype=np.float64)
    testcase.assertTrue(np.isfinite(arr).all(), msg=f"{label} contains non-finite values: {arr}")


def _assert_allclose(
    testcase: unittest.TestCase,
    classical: Any,
    numpyro: Any,
    *,
    label: str,
    rtol: float,
    atol: float = 1e-12,
    rtol_ulps: float | None = None,
    rtol_floor: float = 0.0,
) -> None:
    classical_arr = np.asarray(classical, dtype=np.float64)
    numpyro_native = np.asarray(numpyro)
    numpyro_arr = np.asarray(numpyro_native, dtype=np.float64)
    rtol_eff = float(rtol)
    if rtol_ulps is not None and np.issubdtype(numpyro_native.dtype, np.floating):
        rtol_eff = max(rtol_eff, float(rtol_ulps) * np.finfo(numpyro_native.dtype).eps, float(rtol_floor))
    _assert_all_finite(testcase, classical_arr, label=f"{label} (classical)")
    _assert_all_finite(testcase, numpyro_arr, label=f"{label} (numpyro)")
    np.testing.assert_allclose(classical_arr, numpyro_arr, rtol=rtol_eff, atol=atol, err_msg=label)


@contextmanager
def _jax_x64(enabled: bool):
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _classical_dm_from_params(params: dict[str, float]):
    if {"a", "b", "g"} <= params.keys():
        return ZhaoClassical(
            rs_pc=params["rs_pc"],
            rhos_Msunpc3=params["rhos_Msunpc3"],
            a=params["a"],
            b=params["b"],
            g=params["g"],
            r_t_pc=params["r_t_pc"],
        )
    return NFWClassical(
        rs_pc=params["rs_pc"],
        rhos_Msunpc3=params["rhos_Msunpc3"],
        r_t_pc=params["r_t_pc"],
    )


def _numpyro_dm_from_params(params: dict[str, float]):
    if {"a", "b", "g"} <= params.keys():
        return ZhaoNumPyro()
    return NFWNumPyro()


def _classical_anisotropy_from_kind(kind: str, params: dict[str, float]):
    if kind == "constant":
        return ConstantClassical(beta_ani=params["beta_ani"])
    if kind == "osipkov_merritt":
        return OsipkovMerrittClassical(r_a=params["r_a"])
    if kind == "baes":
        return BaesClassical(
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
    def setUp(self):
        self._jax_x64_ctx = _jax_x64(True)
        self._jax_x64_ctx.__enter__()

    def tearDown(self):
        self._jax_x64_ctx.__exit__(None, None, None)

    def test_plummer_density_matches_classical(self):
        radii = np.geomspace(1e-3, 1e4, 64)
        cases = [
            {"re_pc": 3.0},
            {"re_pc": 200.0},
            {"re_pc": 5e3},
        ]

        model_numpyro = PlummerNumPyro()
        for params in cases:
            with self.subTest(params=params):
                model_classical = PlummerClassical(re_pc=params["re_pc"])
                _assert_allclose(
                    self,
                    model_classical.density_2d(radii),
                    model_numpyro.density_2d(jnp.asarray(radii), re_pc=params["re_pc"]),
                    label=f"Plummer.density_2d {params}",
                    rtol=5e-7,
                    rtol_ulps=16,
                    rtol_floor=5e-15,
                )
                _assert_allclose(
                    self,
                    model_classical.density_3d(radii),
                    model_numpyro.density_3d(jnp.asarray(radii), re_pc=params["re_pc"]),
                    label=f"Plummer.density_3d {params}",
                    rtol=5e-7,
                    rtol_ulps=16,
                    rtol_floor=5e-15,
                )

    def test_nfw_density_and_enclosed_mass_match_classical(self):
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
                model_classical = NFWClassical(**params)
                _assert_allclose(
                    self,
                    model_classical.mass_density_3d(radii),
                    model_numpyro.mass_density_3d(jnp.asarray(radii), params=params),
                    label=f"NFW.mass_density_3d {params}",
                    rtol=0.0,
                    rtol_ulps=16,
                    rtol_floor=5e-15,
                )
                _assert_allclose(
                    self,
                    model_classical.enclosure_mass(radii),
                    model_numpyro.enclosed_mass(jnp.asarray(radii), params=params, method="analytic"),
                    label=f"NFW.enclosed_mass {params}",
                    rtol=case["mass_rtol"],
                    atol=1e-10,
                    rtol_ulps=128,
                    rtol_floor=2e-10,
                )

    def test_zhao_density_and_enclosed_mass_match_classical(self):
        cases = [
            {"rs_pc": 420.0, "rhos_Msunpc3": 0.06, "a": 1.2, "b": 4.0, "g": 0.6, "r_t_pc": 5000.0},
            {"rs_pc": 50.0, "rhos_Msunpc3": 1e-4, "a": 0.5, "b": 6.0, "g": 0.1, "r_t_pc": 1e4},
            {"rs_pc": 2000.0, "rhos_Msunpc3": 5.0, "a": 3.0, "b": 8.0, "g": 1.8, "r_t_pc": 2e4},
        ]

        radii = np.geomspace(1e-3, 1e5, 96)
        model_numpyro = ZhaoNumPyro()
        for params in cases:
            with self.subTest(params=params):
                model_classical = ZhaoClassical(**params)
                _assert_allclose(
                    self,
                    model_classical.mass_density_3d(radii),
                    model_numpyro.mass_density_3d(jnp.asarray(radii), params=params),
                    label=f"Zhao.mass_density_3d {params}",
                    rtol=0.0,
                    rtol_ulps=16,
                    rtol_floor=5e-15,
                )
                _assert_allclose(
                    self,
                    model_classical.enclosure_mass(radii),
                    model_numpyro.enclosed_mass(jnp.asarray(radii), params=params, method="analytic"),
                    label=f"Zhao.enclosed_mass {params}",
                    rtol=0.0,
                    atol=1e-10,
                    rtol_ulps=128,
                    rtol_floor=2e-10,
                )

    def test_constant_anisotropy_methods_match_classical(self):
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
                model_classical = ConstantClassical(**params)
                _assert_allclose(
                    self,
                    model_classical.beta(radii),
                    model_numpyro.beta(jnp.asarray(radii), params=params),
                    label=f"Constant.beta {params}",
                    rtol=0.0,
                    rtol_ulps=16,
                    rtol_floor=5e-15,
                )
                _assert_allclose(
                    self,
                    model_classical.f(radii),
                    model_numpyro.f(jnp.asarray(radii), params=params),
                    label=f"Constant.f {params}",
                    rtol=0.0,
                    rtol_ulps=16,
                    rtol_floor=5e-15,
                )
                _assert_allclose(
                    self,
                    model_classical.kernel(u, R_pc),
                    model_numpyro.kernel(jnp.asarray(u), jnp.asarray(R_pc), params=params),
                    label=f"Constant.kernel {params}",
                    rtol=case["kernel_rtol"],
                    atol=1e-10,
                )

    def test_osipkov_merritt_methods_match_classical(self):
        radii = np.geomspace(0.1, 1e4, 64)
        u = np.geomspace(1.0 + 1e-6, 1e3, 256)
        R_pc = np.array([0.3, 3.0, 300.0], dtype=np.float64)[:, None]

        model_numpyro = OsipkovMerrittNumPyro()
        for r_a in (0.1, 1.0, 350.0, 1e4):
            params = {"r_a": r_a}
            with self.subTest(params=params):
                model_classical = OsipkovMerrittClassical(**params)
                _assert_allclose(
                    self,
                    model_classical.beta(radii),
                    model_numpyro.beta(jnp.asarray(radii), params=params),
                    label=f"OsipkovMerritt.beta {params}",
                    rtol=0.0,
                    rtol_ulps=16,
                    rtol_floor=5e-15,
                )
                _assert_allclose(
                    self,
                    model_classical.f(radii),
                    model_numpyro.f(jnp.asarray(radii), params=params),
                    label=f"OsipkovMerritt.f {params}",
                    rtol=0.0,
                    rtol_ulps=16,
                    rtol_floor=5e-15,
                )
                _assert_allclose(
                    self,
                    model_classical.kernel(u[None, :], R_pc),
                    model_numpyro.kernel(jnp.asarray(u)[None, :], jnp.asarray(R_pc), params=params),
                    label=f"OsipkovMerritt.kernel {params}",
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_baes_methods_match_classical(self):
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
                model_classical = BaesClassical(**params)
                _assert_allclose(
                    self,
                    model_classical.beta(radii),
                    model_numpyro.beta(jnp.asarray(radii), params=params),
                    label=f"Baes.beta {params}",
                    rtol=0.0,
                    rtol_ulps=16,
                    rtol_floor=5e-15,
                )
                _assert_allclose(
                    self,
                    model_classical.f(radii),
                    model_numpyro.f(jnp.asarray(radii), params=params),
                    label=f"Baes.f {params}",
                    rtol=0.0,
                    rtol_ulps=16,
                    rtol_floor=5e-15,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    classical_kernel = model_classical.kernel(u, R_pc, n=320)
                numpyro_kernel = model_numpyro.kernel(
                    jnp.asarray(u)[None, :],
                    jnp.asarray(R_pc)[:, None],
                    params=params,
                    n_kernel=320,
                )
                _assert_allclose(
                    self,
                    classical_kernel,
                    numpyro_kernel,
                    label=f"Baes.kernel {params}",
                    rtol=2e-5,
                    atol=1e-8,
                )

    def test_zhao_enclosed_mass_nfw_limit_is_consistent(self):
        params = {
            "rs_pc": 1200.0,
            "rhos_Msunpc3": 1e-2,
            "a": 1.0,
            "b": 3.0,
            "g": 1.0,
            "r_t_pc": 8000.0,
        }
        radii = np.geomspace(1.0, 1e4, 32)

        classical = ZhaoClassical(**params).enclosure_mass(radii)
        numpyro = ZhaoNumPyro().enclosed_mass(jnp.asarray(radii), params=params, method="analytic")

        _assert_all_finite(self, numpyro, label="Zhao NFW-limit enclosed_mass (numpyro)")
        _assert_all_finite(self, classical, label="Zhao NFW-limit enclosed_mass (classical)")
        _assert_allclose(
            self,
            classical,
            numpyro,
            label="Zhao NFW-limit enclosed_mass",
            rtol=0.0,
            atol=1e-10,
            rtol_ulps=128,
            rtol_floor=2e-10,
        )


class TestDSphConsistencyAgainstClassical(unittest.TestCase):
    def setUp(self):
        self._jax_x64_ctx = _jax_x64(True)
        self._jax_x64_ctx.__enter__()

    def tearDown(self):
        self._jax_x64_ctx.__exit__(None, None, None)

    def _make_classical_dsph(self, params: dict[str, float], anisotropy_kind: str) -> DSphClassical:
        return DSphClassical(
            vmem_kms=params["vmem_kms"],
            submodels={
                "StellarModel": PlummerClassical(re_pc=params["re_pc"]),
                "DMModel": _classical_dm_from_params(params),
                "AnisotropyModel": _classical_anisotropy_from_kind(anisotropy_kind, params),
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

    def test_sigmalos2_integrand_matches_classical(self):
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
                classical_dsph = self._make_classical_dsph(params, case["anisotropy_kind"])
                classical_integrand = classical_dsph.integrand_sigmalos2(u, R_pc, n_kernel=256)
                numpyro_integrand = _numpyro_sigmalos2_integrand(params, case["anisotropy_kind"], u, R_pc)
                _assert_allclose(
                    self,
                    classical_integrand,
                    numpyro_integrand,
                    label=f"DSph.integrand_sigmalos2 {case['name']}",
                    rtol=case["rtol"],
                    atol=1e-10,
                )

    def test_sigmalos2_matches_classical_for_constant_and_osipkov_merritt(self):
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
                classical_dsph = self._make_classical_dsph(params, case["anisotropy_kind"])
                numpyro_dsph = self._make_numpyro_dsph(params, case["anisotropy_kind"])

                classical_sigmalos2 = classical_dsph.sigmalos2_dequad(R_pc, n=2048, n_kernel=256)
                numpyro_sigmalos2 = numpyro_dsph.sigmalos2(
                    jnp.asarray(R_pc),
                    params=params,
                    backend="kernel",
                    n_u=1024,
                    u_max=5000.0,
                )
                _assert_allclose(
                    self,
                    classical_sigmalos2,
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

        classical_dsph = self._make_classical_dsph(params, "baes")
        numpyro_dsph = self._make_numpyro_dsph(params, "baes")

        classical_sigmalos2 = classical_dsph.sigmalos2_dequad(R_pc, n=1024, n_kernel=128)
        numpyro_sigmalos2 = numpyro_dsph.sigmalos2(
            jnp.asarray(R_pc),
            params=params,
            backend="kernel",
            n_u=768,
            u_max=3000.0,
        )
        _assert_allclose(
            self,
            classical_sigmalos2,
            numpyro_sigmalos2,
            label="DSph.sigmalos2 baes",
            rtol=1e-2,
            atol=1e-8,
        )

    def test_classical_sigmalos2_uses_the_shared_entry_point(self):
        params = {
            "re_pc": 200.0,
            "rs_pc": 1200.0,
            "rhos_Msunpc3": 1e-2,
            "r_t_pc": 8000.0,
            "beta_ani": 0.2,
            "vmem_kms": 0.0,
        }
        classical_dsph = self._make_classical_dsph(params, "constant")
        R_pc = np.array([50.0, 100.0])

        with patch.object(
            classical_dsph,
            "sigmalos2_dequad",
            return_value=np.array([1.0, 2.0]),
        ) as dequad:
            result = classical_dsph.sigmalos2(
                R_pc,
                n=64,
                n_kernel=32,
                ignore_RuntimeWarning=False,
            )

        dequad.assert_called_once_with(R_pc, 64, 32, False)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
