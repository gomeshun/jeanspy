#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import emcee
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "benchmarks" / "jeans_comparison" / "artifacts"
REPORT_PATH = ROOT / "benchmarks" / "jeans_comparison" / "README.md"

PC_PER_ARCSEC_DISTANCE_MPC = 0.648 / math.pi
G_KMS2_PC_PER_MSUN = 0.004300917270036279

TRUE_PARAMS: dict[str, float] = {
    "re_pc": 200.0,
    "rs_pc": 1200.0,
    "rhos_Msunpc3": 1.0e-2,
    "r_t_pc": 1.0e8,
    "r_a_pc": 600.0,
    "vmem_kms": 0.0,
}

PRIOR_BOUNDS: dict[str, tuple[float, float]] = {
    "log10_rs_pc": (2.0, 4.3),
    "log10_rhos_Msunpc3": (-4.0, -0.3),
    "log10_r_a_pc": (1.5, 4.3),
}


@dataclass(frozen=True)
class ProfileData:
    radius_pc: np.ndarray
    sigma_kms: np.ndarray
    sigma_err_kms: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class Engine:
    name: str
    status: str
    predict_sigma: Callable[[np.ndarray], np.ndarray] | None
    setup_seconds: float
    note: str = ""
    sampler_kind: str = "emcee"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _import_optional(module: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(module), None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"{type(exc).__name__}: {exc}"


def agama_mass_from_nfw_params(rs_pc: float, rhos_msunpc3: float) -> float:
    """Return the AGAMA NFW normalization for JeansPy NFW parameters."""
    return 4.0 * math.pi * rhos_msunpc3 * rs_pc**3


def write_create_mock_config(
    path: Path,
    *,
    output_snapshot: Path,
    anisotropy: str,
    n_stars: int,
    seed: int,
    true_params: dict[str, float],
) -> None:
    anisotropy_block = 'model = "isotropic"'
    if anisotropy == "om":
        anisotropy_block = f'model = "osipkov_merritt"\nr_a = {true_params["r_a_pc"]:.16g}'

    agama_nfw_mass = agama_mass_from_nfw_params(
        true_params["rs_pc"], true_params["rhos_Msunpc3"]
    )
    text = f"""[units]
mass = 1.0
length = 0.001
velocity = 1.0

[stellar_density]
model = "plummer"
mass = 100000.0
scale_radius = {true_params["re_pc"]:.16g}

[dark_matter_potential]
model = "nfw"
mass = {agama_nfw_mass:.16g}
scale_radius = {true_params["rs_pc"]:.16g}

[anisotropy]
{anisotropy_block}

[sampling]
n_particles = {int(n_stars)}
seed = {int(seed)}
method = 0

[output]
path = "{output_snapshot.as_posix()}"
format = "text"
"""
    path.write_text(text, encoding="utf-8")


def generate_mock_with_create_mock(
    *,
    output_dir: Path,
    anisotropy: str,
    n_stars: int,
    seed: int,
    velocity_error_kms: float,
    true_params: dict[str, float],
) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "external"))
    app = importlib.import_module("create_mock.main")

    config_path = output_dir / f"mock_{anisotropy}_{n_stars}_seed{seed}.toml"
    snapshot_path = output_dir / f"mock_{anisotropy}_{n_stars}_seed{seed}_snapshot.dat"
    write_create_mock_config(
        config_path,
        output_snapshot=snapshot_path,
        anisotropy=anisotropy,
        n_stars=n_stars,
        seed=seed,
        true_params=true_params,
    )

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        app.run(config_path)

    data = np.loadtxt(snapshot_path, ndmin=2)
    rng = np.random.default_rng(seed + 1009)
    R_pc = np.hypot(data[:, 0], data[:, 1])
    vlos_true = data[:, 5] + true_params["vmem_kms"]
    e_vlos = np.full(R_pc.size, velocity_error_kms, dtype=float)
    vlos_obs = vlos_true + rng.normal(0.0, e_vlos)

    npz_path = output_dir / f"mock_{anisotropy}_{n_stars}_seed{seed}.npz"
    np.savez_compressed(
        npz_path,
        R_pc=R_pc,
        vlos_kms=vlos_obs,
        vlos_true_kms=vlos_true,
        e_vlos_kms=e_vlos,
        phase_space=data[:, :6],
        masses=data[:, 6],
        true_params=json.dumps(true_params),
        anisotropy=anisotropy,
        source="create_mock",
        config_path=str(config_path),
        snapshot_path=str(snapshot_path),
    )
    return {
        "path": npz_path,
        "source": "create_mock",
        "config_path": config_path,
        "snapshot_path": snapshot_path,
        "R_pc": R_pc,
        "vlos_kms": vlos_obs,
        "e_vlos_kms": e_vlos,
    }


def generate_mock_with_jeanspy_fallback(
    *,
    output_dir: Path,
    anisotropy: str,
    n_stars: int,
    seed: int,
    velocity_error_kms: float,
    true_params: dict[str, float],
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from jeanspy.model_numpyro import (
        ConstantAnisotropyModel,
        DSphModel,
        NFWModel,
        OsipkovMerrittModel,
        PlummerModel,
    )

    key = jax.random.PRNGKey(seed)
    key, r_key, v_key, e_key = jax.random.split(key, 4)
    stellar = PlummerModel()
    R_pc = np.asarray(stellar.sample_R(r_key, n_stars, re_pc=true_params["re_pc"]))
    ani_model = ConstantAnisotropyModel() if anisotropy == "isotropic" else OsipkovMerrittModel()
    dsph = DSphModel(
        submodels={"StellarModel": stellar, "DMModel": NFWModel(), "AnisotropyModel": ani_model}
    )
    params = {
        "re_pc": true_params["re_pc"],
        "rs_pc": true_params["rs_pc"],
        "rhos_Msunpc3": true_params["rhos_Msunpc3"],
        "r_t_pc": true_params["r_t_pc"],
        "vmem_kms": true_params["vmem_kms"],
    }
    if anisotropy == "isotropic":
        params["beta_ani"] = 0.0
    else:
        params["r_a"] = true_params["r_a_pc"]
    sigma2 = np.asarray(
        dsph.sigmalos2(
            jnp.asarray(R_pc),
            params=params,
            backend="kernel",
            jit=True,
            n_u=192,
            u_max=3000.0,
            constant_kernel_backend="jax",
        )
    )
    e_vlos = np.full(R_pc.size, velocity_error_kms, dtype=float)
    vlos_true = true_params["vmem_kms"] + np.sqrt(np.maximum(sigma2, 0.0)) * np.asarray(
        jax.random.normal(v_key, shape=(n_stars,))
    )
    vlos_obs = vlos_true + np.asarray(jax.random.normal(e_key, shape=(n_stars,))) * e_vlos
    npz_path = output_dir / f"mock_{anisotropy}_{n_stars}_seed{seed}_fallback.npz"
    np.savez_compressed(
        npz_path,
        R_pc=R_pc,
        vlos_kms=vlos_obs,
        vlos_true_kms=vlos_true,
        e_vlos_kms=e_vlos,
        true_params=json.dumps(true_params),
        anisotropy=anisotropy,
        source="jeanspy_fallback",
    )
    return {
        "path": npz_path,
        "source": "jeanspy_fallback",
        "R_pc": R_pc,
        "vlos_kms": vlos_obs,
        "e_vlos_kms": e_vlos,
    }


def generate_mock(
    *,
    output_dir: Path,
    anisotropy: str,
    n_stars: int,
    seed: int,
    velocity_error_kms: float,
    true_params: dict[str, float],
    source: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if source in {"auto", "create_mock"}:
        try:
            return generate_mock_with_create_mock(
                output_dir=output_dir,
                anisotropy=anisotropy,
                n_stars=n_stars,
                seed=seed,
                velocity_error_kms=velocity_error_kms,
                true_params=true_params,
            )
        except Exception as exc:
            if source == "create_mock":
                raise
            print(f"[mock] create_mock failed; using jeanspy fallback: {type(exc).__name__}: {exc}")
    return generate_mock_with_jeanspy_fallback(
        output_dir=output_dir,
        anisotropy=anisotropy,
        n_stars=n_stars,
        seed=seed,
        velocity_error_kms=velocity_error_kms,
        true_params=true_params,
    )


def make_dispersion_profile(
    R_pc: np.ndarray,
    vlos_kms: np.ndarray,
    *,
    n_bins: int,
) -> ProfileData:
    order = np.argsort(R_pc)
    R_sorted = np.asarray(R_pc, dtype=float)[order]
    v_sorted = np.asarray(vlos_kms, dtype=float)[order]
    chunks = [chunk for chunk in np.array_split(np.arange(R_sorted.size), n_bins) if chunk.size >= 6]

    radii: list[float] = []
    sigma: list[float] = []
    sigma_err: list[float] = []
    counts: list[int] = []
    for chunk in chunks:
        R_bin = R_sorted[chunk]
        v_bin = v_sorted[chunk]
        sig = float(np.std(v_bin, ddof=1))
        if not np.isfinite(sig) or sig <= 0:
            continue
        radii.append(float(np.exp(np.mean(np.log(np.clip(R_bin, 1e-12, None))))))
        sigma.append(sig)
        sigma_err.append(sig / math.sqrt(2.0 * (v_bin.size - 1)))
        counts.append(int(v_bin.size))

    return ProfileData(
        radius_pc=np.asarray(radii),
        sigma_kms=np.asarray(sigma),
        sigma_err_kms=np.asarray(sigma_err),
        counts=np.asarray(counts, dtype=int),
    )


def theta_names(anisotropy: str) -> list[str]:
    names = ["log10_rs_pc", "log10_rhos_Msunpc3"]
    if anisotropy == "om":
        names.append("log10_r_a_pc")
    return names


def stable_seed_offset(name: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(name)) % 10000


def theta_to_params(theta: np.ndarray, anisotropy: str) -> dict[str, float]:
    names = theta_names(anisotropy)
    values = dict(zip(names, np.asarray(theta, dtype=float), strict=True))
    params = {
        "rs_pc": 10.0 ** values["log10_rs_pc"],
        "rhos_Msunpc3": 10.0 ** values["log10_rhos_Msunpc3"],
    }
    if anisotropy == "om":
        params["r_a_pc"] = 10.0 ** values["log10_r_a_pc"]
    return params


def params_to_theta(params: dict[str, float], anisotropy: str) -> np.ndarray:
    values = [math.log10(params["rs_pc"]), math.log10(params["rhos_Msunpc3"])]
    if anisotropy == "om":
        values.append(math.log10(params["r_a_pc"]))
    return np.asarray(values, dtype=float)


def log_prior(theta: np.ndarray, anisotropy: str) -> float:
    for name, value in zip(theta_names(anisotropy), theta, strict=True):
        lo, hi = PRIOR_BOUNDS[name]
        if not lo <= value <= hi:
            return -np.inf
    return 0.0


def log_likelihood_profile(
    theta: np.ndarray,
    *,
    anisotropy: str,
    profile: ProfileData,
    predict_sigma: Callable[[np.ndarray], np.ndarray],
) -> float:
    sigma_model = np.asarray(predict_sigma(theta), dtype=float)
    if sigma_model.shape != profile.sigma_kms.shape or not np.all(np.isfinite(sigma_model)):
        return -np.inf
    resid = (profile.sigma_kms - sigma_model) / profile.sigma_err_kms
    return float(
        -0.5 * np.sum(resid**2 + np.log(2.0 * np.pi * profile.sigma_err_kms**2))
    )


def log_probability(
    theta: np.ndarray,
    *,
    anisotropy: str,
    profile: ProfileData,
    predict_sigma: Callable[[np.ndarray], np.ndarray],
) -> float:
    lp = log_prior(theta, anisotropy)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_profile(
        theta,
        anisotropy=anisotropy,
        profile=profile,
        predict_sigma=predict_sigma,
    )
    return lp + ll


def initial_walkers(
    *,
    anisotropy: str,
    n_walkers: int,
    seed: int,
    true_params: dict[str, float],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    center = params_to_theta(true_params, anisotropy)
    scales = np.full(center.size, 0.04)
    walkers = center + rng.normal(0.0, scales, size=(n_walkers, center.size))
    for i, name in enumerate(theta_names(anisotropy)):
        lo, hi = PRIOR_BOUNDS[name]
        walkers[:, i] = np.clip(walkers[:, i], lo + 1e-5, hi - 1e-5)
    return walkers


def make_jeanspy_engine(anisotropy: str, R_pc: np.ndarray, args: argparse.Namespace) -> Engine:
    start = time.perf_counter()
    try:
        import jax
        import jax.numpy as jnp

        from jeanspy.model_numpyro import (
            ConstantAnisotropyModel,
            DSphModel,
            NFWModel,
            OsipkovMerrittModel,
            PlummerModel,
        )
    except Exception as exc:
        return Engine("jeanspy", "skipped", None, 0.0, f"import failed: {exc}")

    ani = ConstantAnisotropyModel() if anisotropy == "isotropic" else OsipkovMerrittModel()
    dsph = DSphModel(
        submodels={"StellarModel": PlummerModel(), "DMModel": NFWModel(), "AnisotropyModel": ani}
    )
    R_jax = jnp.asarray(R_pc)

    def predict(theta: np.ndarray) -> np.ndarray:
        params_fit = theta_to_params(theta, anisotropy)
        params: dict[str, Any] = {
            "re_pc": TRUE_PARAMS["re_pc"],
            "rs_pc": params_fit["rs_pc"],
            "rhos_Msunpc3": params_fit["rhos_Msunpc3"],
            "r_t_pc": TRUE_PARAMS["r_t_pc"],
        }
        if anisotropy == "isotropic":
            params["beta_ani"] = 0.0
        else:
            params["r_a"] = params_fit["r_a_pc"]
        sigma2 = dsph.sigmalos2(
            R_jax,
            params=params,
            backend="kernel",
            jit=True,
            n_u=args.jeanspy_n_u,
            u_max=args.jeanspy_u_max,
            constant_kernel_backend="jax",
        )
        return np.sqrt(np.maximum(np.asarray(jax.block_until_ready(sigma2)), 0.0))

    predict(params_to_theta(TRUE_PARAMS, anisotropy))
    return Engine("jeanspy", "ok", predict, time.perf_counter() - start)


def make_jeanspy_nuts_engine(anisotropy: str, R_pc: np.ndarray, args: argparse.Namespace) -> Engine:
    engine = make_jeanspy_engine(anisotropy, R_pc, args)
    return Engine(
        "jeanspy-nuts",
        engine.status,
        engine.predict_sigma,
        engine.setup_seconds,
        engine.note,
        sampler_kind="nuts",
    )


def _fit_plummer_surface_mge(re_pc: float, ngauss: int) -> tuple[np.ndarray, np.ndarray]:
    import mgefit as mge

    R = np.geomspace(re_pc * 1e-3, re_pc * 1e3, 500)
    surface = 1.0 / (math.pi * re_pc**2) / (1.0 + (R / re_pc) ** 2) ** 2
    fit = mge.fit_1d(
        R,
        surface,
        ngauss=ngauss,
        inner_slope=0,
        outer_slope=4,
        linear=False,
        quiet=True,
        plot=False,
    )
    counts, sigma = fit.sol
    surf = counts / (math.sqrt(2.0 * math.pi) * sigma)
    return np.asarray(surf, dtype=float), np.asarray(sigma, dtype=float)


def _fit_nfw_density_mge(ngauss: int) -> tuple[np.ndarray, np.ndarray]:
    import mgefit as mge

    x = np.geomspace(1e-4, 1e4, 700)
    rho = 1.0 / (x * (1.0 + x) ** 2)
    fit = mge.fit_1d(
        x,
        rho,
        ngauss=ngauss,
        inner_slope=1,
        outer_slope=3,
        linear=False,
        quiet=True,
        plot=False,
    )
    surf_dimensionless, sigma_dimensionless = fit.sol
    return np.asarray(surf_dimensionless, dtype=float), np.asarray(sigma_dimensionless, dtype=float)


def make_jampy_engine(anisotropy: str, R_pc: np.ndarray, args: argparse.Namespace) -> Engine:
    start = time.perf_counter()
    jam, err = _import_optional("jampy")
    if jam is None:
        return Engine("jampy", "skipped", None, 0.0, f"import failed: {err}")
    mgefit, err = _import_optional("mgefit")
    if mgefit is None:
        return Engine("jampy", "skipped", None, 0.0, f"mgefit import failed: {err}")

    surf_lum, sigma_lum = _fit_plummer_surface_mge(TRUE_PARAMS["re_pc"], args.jampy_tracer_ngauss)
    nfw_surf_unit, nfw_sigma_unit = _fit_nfw_density_mge(args.jampy_nfw_ngauss)

    def predict(theta: np.ndarray) -> np.ndarray:
        params_fit = theta_to_params(theta, anisotropy)
        rs_pc = params_fit["rs_pc"]
        rhos = params_fit["rhos_Msunpc3"]
        surf_pot = nfw_surf_unit * rhos * rs_pc
        sigma_pot = nfw_sigma_unit * rs_pc
        beta = np.zeros_like(surf_lum)
        rani = None
        if anisotropy == "om":
            rani = params_fit["r_a_pc"]
        out = jam.sph.proj(
            surf_lum,
            sigma_lum,
            surf_pot,
            sigma_pot,
            0.0,
            PC_PER_ARCSEC_DISTANCE_MPC,
            R_pc,
            beta=beta,
            rani=rani,
            sigmapsf=0.0,
            pixsize=0.0,
            plot=False,
            quiet=True,
            epsrel=args.jampy_epsrel,
            tensor="los",
        )
        return np.asarray(out.model, dtype=float)

    predict(params_to_theta(TRUE_PARAMS, anisotropy))
    return Engine("jampy", "ok", predict, time.perf_counter() - start)


def make_agama_engine(anisotropy: str, R_pc: np.ndarray, args: argparse.Namespace) -> Engine:
    start = time.perf_counter()
    agama, err = _import_optional("agama")
    if agama is None:
        return Engine("agama", "skipped", None, 0.0, f"import failed: {err}")

    agama.setUnits(mass=1.0, length=0.001, velocity=1.0)
    tracer = agama.Density(type="Plummer", mass=1.0e5, scaleRadius=TRUE_PARAMS["re_pc"])
    xy = np.column_stack([R_pc, np.zeros_like(R_pc)])

    def predict(theta: np.ndarray) -> np.ndarray:
        params_fit = theta_to_params(theta, anisotropy)
        halo = agama.Potential(
            type="NFW",
            mass=agama_mass_from_nfw_params(params_fit["rs_pc"], params_fit["rhos_Msunpc3"]),
            scaleRadius=params_fit["rs_pc"],
        )
        df_kwargs: dict[str, Any] = {
            "type": "QuasiSpherical",
            "density": tracer,
            "potential": halo,
            "beta0": 0.0,
        }
        if anisotropy == "om":
            df_kwargs["r_a"] = params_fit["r_a_pc"]
        df = agama.DistributionFunction(**df_kwargs)
        gm = agama.GalaxyModel(halo, df)
        _, mean, vel2 = gm.moments(xy, dens=True, vel=True, vel2=True)
        los2 = np.asarray(vel2)[:, 2] - np.asarray(mean)[:, 2] ** 2
        return np.sqrt(np.maximum(los2, 0.0))

    predict(params_to_theta(TRUE_PARAMS, anisotropy))
    return Engine("agama", "ok", predict, time.perf_counter() - start)


def _load_gravsphere_v1_functions() -> tuple[Any | None, str | None]:
    source_dir = ROOT / "external" / "gravsphere"
    source_path = source_dir / "functions.py"
    if not source_path.exists():
        return None, "external/gravsphere submodule is not present"

    try:
        import scipy.integrate

        def _simps(y: np.ndarray, x: np.ndarray | None = None, **kwargs: Any) -> Any:
            return scipy.integrate.simpson(y, x=x, **kwargs)

        scipy.integrate.simps = _simps  # type: ignore[attr-defined]

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                import scipy.misc.common as scipy_misc_common
        except Exception:
            import types

            scipy_misc_common = types.ModuleType("scipy.misc.common")
            sys.modules["scipy.misc.common"] = scipy_misc_common

        if not hasattr(scipy_misc_common, "derivative"):

            def _derivative(
                func: Callable[..., Any],
                x0: Any,
                dx: float = 1.0,
                n: int = 1,
                order: int = 3,
                args: tuple[Any, ...] = (),
            ) -> Any:
                if n != 1:
                    raise NotImplementedError("legacy gravsphere adapter only needs first derivatives")
                return (func(x0 + dx, *args) - func(x0 - dx, *args)) / (2.0 * dx)

            scipy_misc_common.derivative = _derivative

        if not hasattr(np, "int"):
            np.int = int  # type: ignore[attr-defined]

        module_name = "_jeans_benchmark_gravsphere_v1_functions"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            return None, f"could not load {source_path}"

        old_path = list(sys.path)
        sys.path.insert(0, str(source_dir))
        try:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = old_path
        return module, None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"{type(exc).__name__}: {exc}"


def make_gravsphere_v1_engine(anisotropy: str, R_pc: np.ndarray, _args: argparse.Namespace) -> Engine:
    start = time.perf_counter()
    gs, err = _load_gravsphere_v1_functions()
    if gs is None:
        return Engine("gravsphere-v1", "skipped", None, 0.0, f"import failed: {err}")

    R_kpc = np.asarray(R_pc, dtype=float) / 1000.0
    re_kpc = TRUE_PARAMS["re_pc"] / 1000.0
    mstar_rad = np.asarray([1.0e-8, 1.0], dtype=float)
    mstar_prof = np.asarray([0.0, 0.0], dtype=float)

    def plummer_nu(r: np.ndarray, nupars: np.ndarray) -> np.ndarray:
        a = nupars[0]
        return 3.0 / (4.0 * math.pi * a**3) * (1.0 + (r / a) ** 2) ** -2.5

    def plummer_sigma(r: np.ndarray, nupars: np.ndarray) -> np.ndarray:
        a = nupars[0]
        return 1.0 / (math.pi * a**2) * (1.0 + (r / a) ** 2) ** -2.0

    def nfw_mass(r: np.ndarray, mpars: np.ndarray) -> np.ndarray:
        rs_kpc, rhos_msun_kpc3 = mpars
        x = np.asarray(r, dtype=float) / rs_kpc
        return 4.0 * math.pi * rhos_msun_kpc3 * rs_kpc**3 * (np.log1p(x) - x / (1.0 + x))

    def no_central_mass(r: np.ndarray, _mpars: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(r, dtype=float))

    if anisotropy == "isotropic":

        def beta_func(r: np.ndarray, _betpars: np.ndarray) -> np.ndarray:
            return np.zeros_like(np.asarray(r, dtype=float))

        def beta_factor(r: np.ndarray, _betpars: np.ndarray, _rhalf: float, _arot: float) -> np.ndarray:
            return np.ones_like(np.asarray(r, dtype=float))

        beta_pars = np.asarray([np.inf], dtype=float)
    else:

        def beta_func(r: np.ndarray, betpars: np.ndarray) -> np.ndarray:
            r_arr = np.asarray(r, dtype=float)
            r_a = betpars[0]
            return r_arr**2 / (r_arr**2 + r_a**2)

        def beta_factor(r: np.ndarray, betpars: np.ndarray, _rhalf: float, _arot: float) -> np.ndarray:
            r_arr = np.asarray(r, dtype=float)
            return r_arr**2 + betpars[0] ** 2

        beta_pars = np.asarray([TRUE_PARAMS["r_a_pc"] / 1000.0], dtype=float)

    def predict(theta: np.ndarray) -> np.ndarray:
        params_fit = theta_to_params(theta, anisotropy)
        rs_kpc = params_fit["rs_pc"] / 1000.0
        rhos_msun_kpc3 = params_fit["rhos_Msunpc3"] * 1.0e9
        if anisotropy == "om":
            beta_pars[0] = params_fit["r_a_pc"] / 1000.0
        rmin = max(float(np.min(R_kpc[R_kpc > 0])) * 1.0e-2, 1.0e-6)
        rmax = max(float(np.max(R_kpc)) * 20.0, 50.0 * rs_kpc, 100.0 * re_kpc)
        _, _, los2 = gs.sigp(
            R_kpc,
            R_kpc,
            plummer_nu,
            plummer_sigma,
            nfw_mass,
            no_central_mass,
            beta_func,
            beta_factor,
            np.asarray([re_kpc], dtype=float),
            np.asarray([rs_kpc, rhos_msun_kpc3], dtype=float),
            beta_pars,
            mstar_rad,
            mstar_prof,
            0.0,
            0.0,
            re_kpc,
            gs.Guse,
            rmin,
            rmax,
        )
        return np.sqrt(np.maximum(np.asarray(los2, dtype=float), 0.0)) / 1000.0

    predict(params_to_theta(TRUE_PARAMS, anisotropy))
    note = "official legacy GravSphere second-moment sigp adapter; sampler is the common benchmark emcee wrapper"
    return Engine("gravsphere-v1", "ok", predict, time.perf_counter() - start, note)


def make_gravsphere2_engine(_anisotropy: str, _R_pc: np.ndarray, _args: argparse.Namespace) -> Engine:
    note = (
        "skipped: LOS-only mode still calls sigp_k and lnpdfj, so the native "
        "workflow fits a second+fourth-moment velocity PDF rather than a Gaussian "
        "second-moment-only profile likelihood."
    )
    return Engine("gravsphere2", "skipped", None, 0.0, note)


def make_pygravsphere_engine(_anisotropy: str, _R_pc: np.ndarray, _args: argparse.Namespace) -> Engine:
    note = (
        "skipped: pyGravSphere exposes a fast SWIG/C second-moment API, but the "
        "published wrapper targets legacy Python/SWIG builds and is not installable "
        "in this Python 3.12 uv environment without a rebuild/port."
    )
    return Engine("pygravsphere", "skipped", None, 0.0, note)


ENGINE_FACTORIES: dict[str, Callable[[str, np.ndarray, argparse.Namespace], Engine]] = {
    "jeanspy": make_jeanspy_engine,
    "jeanspy-emcee": make_jeanspy_engine,
    "jeanspy-nuts": make_jeanspy_nuts_engine,
    "jampy": make_jampy_engine,
    "agama": make_agama_engine,
    "gravsphere-v1": make_gravsphere_v1_engine,
    "gravsphere2": make_gravsphere2_engine,
    "pygravsphere": make_pygravsphere_engine,
}


def estimate_prediction_seconds(predict: Callable[[np.ndarray], np.ndarray], theta0: np.ndarray) -> float:
    times: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        predict(theta0)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def run_engine_sampler(
    *,
    engine: Engine,
    anisotropy: str,
    profile: ProfileData,
    args: argparse.Namespace,
) -> dict[str, Any]:
    assert engine.predict_sigma is not None
    if engine.sampler_kind == "nuts":
        return run_jeanspy_nuts_sampler(
            engine=engine,
            anisotropy=anisotropy,
            profile=profile,
            args=args,
        )

    names = theta_names(anisotropy)
    ndim = len(names)
    n_walkers = max(args.n_walkers, 2 * ndim + 2)
    p0 = initial_walkers(
        anisotropy=anisotropy,
        n_walkers=n_walkers,
        seed=args.seed + stable_seed_offset(engine.name),
        true_params=TRUE_PARAMS,
    )
    theta0 = params_to_theta(TRUE_PARAMS, anisotropy)
    pred_seconds = estimate_prediction_seconds(engine.predict_sigma, theta0)
    if pred_seconds > args.slow_eval_threshold and not args.include_slow:
        return {
            "engine": engine.name,
            "status": "skipped",
            "note": (
                f"median prediction time {pred_seconds:.3g}s exceeds "
                f"--slow-eval-threshold={args.slow_eval_threshold}; rerun with --include-slow"
            ),
            "setup_seconds": engine.setup_seconds,
            "prediction_seconds": pred_seconds,
        }

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        log_probability,
        kwargs={
            "anisotropy": anisotropy,
            "profile": profile,
            "predict_sigma": engine.predict_sigma,
        },
    )
    start = time.perf_counter()
    sampler.run_mcmc(p0, args.n_steps, progress=args.progress)
    elapsed = time.perf_counter() - start
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    discard = min(args.n_burn, max(args.n_steps // 2, 0))
    flat = sampler.get_chain(discard=discard, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=discard, flat=True)

    out_path = args.output_dir / f"posterior_{engine.name}_{anisotropy}_n{args.n_stars}_seed{args.seed}.npz"
    np.savez_compressed(
        out_path,
        chain=chain,
        log_prob=log_prob,
        flat_chain=flat,
        flat_log_prob=flat_log_prob,
        theta_names=np.asarray(names),
        anisotropy=anisotropy,
        engine=engine.name,
    )

    finite = np.isfinite(flat_log_prob)
    if np.any(finite):
        best_index = int(np.argmax(flat_log_prob))
        best_theta = flat[best_index]
    else:
        best_theta = np.full(ndim, np.nan)

    quantiles: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        values = flat[:, i]
        quantiles[name] = {
            "q16": float(np.nanpercentile(values, 16)),
            "q50": float(np.nanpercentile(values, 50)),
            "q84": float(np.nanpercentile(values, 84)),
        }

    acceptance = float(np.mean(sampler.acceptance_fraction))
    return {
        "engine": engine.name,
        "status": "ok",
        "note": engine.note,
        "sampler": "emcee",
        "setup_seconds": engine.setup_seconds,
        "prediction_seconds": pred_seconds,
        "sampling_seconds": elapsed,
        "n_walkers": n_walkers,
        "n_steps": args.n_steps,
        "n_burn": discard,
        "n_posterior_samples": int(flat.shape[0]),
        "acceptance_fraction": acceptance,
        "theta_names": names,
        "posterior_path": out_path,
        "best_theta": best_theta,
        "best_params": theta_to_params(best_theta, anisotropy) if np.all(np.isfinite(best_theta)) else {},
        "quantiles": quantiles,
    }


def run_jeanspy_nuts_sampler(
    *,
    engine: Engine,
    anisotropy: str,
    profile: ProfileData,
    args: argparse.Namespace,
) -> dict[str, Any]:
    assert engine.predict_sigma is not None

    pred_seconds = estimate_prediction_seconds(
        engine.predict_sigma,
        params_to_theta(TRUE_PARAMS, anisotropy),
    )

    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    from jeanspy.model_numpyro import (
        ConstantAnisotropyModel,
        DSphModel,
        NFWModel,
        OsipkovMerrittModel,
        PlummerModel,
    )

    ani = ConstantAnisotropyModel() if anisotropy == "isotropic" else OsipkovMerrittModel()
    dsph = DSphModel(
        submodels={"StellarModel": PlummerModel(), "DMModel": NFWModel(), "AnisotropyModel": ani}
    )
    R_jax = jnp.asarray(profile.radius_pc)
    sigma_obs = jnp.asarray(profile.sigma_kms)
    sigma_err = jnp.asarray(profile.sigma_err_kms)

    def numpyro_model() -> None:
        sampled: dict[str, Any] = {}
        for name in theta_names(anisotropy):
            lo, hi = PRIOR_BOUNDS[name]
            sampled[name] = numpyro.sample(name, dist.Uniform(lo, hi))

        params: dict[str, Any] = {
            "re_pc": TRUE_PARAMS["re_pc"],
            "rs_pc": jnp.power(10.0, sampled["log10_rs_pc"]),
            "rhos_Msunpc3": jnp.power(10.0, sampled["log10_rhos_Msunpc3"]),
            "r_t_pc": TRUE_PARAMS["r_t_pc"],
        }
        if anisotropy == "isotropic":
            params["beta_ani"] = 0.0
        else:
            params["r_a"] = jnp.power(10.0, sampled["log10_r_a_pc"])

        sigma2 = dsph.sigmalos2(
            R_jax,
            params=params,
            backend="kernel",
            jit=True,
            n_u=args.jeanspy_n_u,
            u_max=args.jeanspy_u_max,
            constant_kernel_backend="jax",
        )
        sigma_model = jnp.sqrt(jnp.clip(sigma2, min=1e-12, max=1e12))
        numpyro.sample("sigma_obs", dist.Normal(sigma_model, sigma_err), obs=sigma_obs)

    kernel = NUTS(numpyro_model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.nuts_warmup,
        num_samples=args.nuts_samples,
        num_chains=args.nuts_chains,
        progress_bar=args.progress,
    )
    start = time.perf_counter()
    rng_key = jax.random.PRNGKey(args.seed + 9001)
    if args.nuts_chains > 1:
        rng_key = jax.random.split(rng_key, args.nuts_chains)
    mcmc.run(rng_key, extra_fields=("accept_prob", "diverging"))
    elapsed = time.perf_counter() - start

    names = theta_names(anisotropy)
    samples_grouped = mcmc.get_samples(group_by_chain=True)
    samples_flat = mcmc.get_samples(group_by_chain=False)
    chain = np.stack([np.asarray(samples_grouped[name]) for name in names], axis=-1)
    flat = np.column_stack([np.asarray(samples_flat[name]) for name in names])
    flat_log_prob = np.full(flat.shape[0], np.nan)

    extra = mcmc.get_extra_fields()
    accept_prob = extra.get("accept_prob") if isinstance(extra, dict) else None
    diverging = extra.get("diverging") if isinstance(extra, dict) else None
    acceptance = float(np.mean(np.asarray(accept_prob))) if accept_prob is not None else None
    n_divergent = int(np.sum(np.asarray(diverging))) if diverging is not None else None

    out_path = args.output_dir / f"posterior_{engine.name}_{anisotropy}_n{args.n_stars}_seed{args.seed}.npz"
    np.savez_compressed(
        out_path,
        chain=chain,
        flat_chain=flat,
        flat_log_prob=flat_log_prob,
        theta_names=np.asarray(names),
        anisotropy=anisotropy,
        engine=engine.name,
        sampler="nuts",
    )

    quantiles: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        values = flat[:, i]
        quantiles[name] = {
            "q16": float(np.nanpercentile(values, 16)),
            "q50": float(np.nanpercentile(values, 50)),
            "q84": float(np.nanpercentile(values, 84)),
        }

    med_theta = np.asarray([quantiles[name]["q50"] for name in names], dtype=float)
    return {
        "engine": engine.name,
        "status": "ok",
        "note": engine.note,
        "sampler": "nuts",
        "setup_seconds": engine.setup_seconds,
        "prediction_seconds": pred_seconds,
        "sampling_seconds": elapsed,
        "n_warmup": args.nuts_warmup,
        "n_samples": args.nuts_samples,
        "n_chains": args.nuts_chains,
        "n_posterior_samples": int(flat.shape[0]),
        "acceptance_fraction": acceptance,
        "n_divergent": n_divergent,
        "theta_names": names,
        "posterior_path": out_path,
        "best_theta": med_theta,
        "best_params": theta_to_params(med_theta, anisotropy),
        "quantiles": quantiles,
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Jeans Code Benchmark")
    lines.append("")
    lines.append("Generated by `scripts/benchmark_jeans_codes.py`.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Anisotropy: `{summary['anisotropy']}`")
    lines.append(f"- Mock source: `{summary['mock']['source']}`")
    lines.append(f"- Stars: `{summary['n_stars']}`")
    lines.append(f"- Bins: `{summary['n_bins_effective']}`")
    lines.append(
        f"- emcee sampler: `{summary['sampler']['n_walkers']}` walkers and `{summary['sampler']['n_steps']}` steps"
    )
    lines.append(
        f"- NUTS sampler: `{summary['sampler']['nuts_warmup']}` warmup, `{summary['sampler']['nuts_samples']}` samples, `{summary['sampler']['nuts_chains']}` chain(s)"
    )
    lines.append("- Fixed Plummer projected half-light radius: `re_pc = 200 pc`")
    lines.append("- Priors: log-flat in `rs_pc`, `rhos_Msunpc3`, and `r_a_pc` when OM is used.")
    lines.append("")
    lines.append("True parameters:")
    for key, value in TRUE_PARAMS.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Engine Results")
    lines.append("")
    lines.append(
        "| Engine | Sampler | Status | Sampling s | Median predict s | Acceptance | Divergences | Posterior | Note |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---|")
    for result in summary["results"]:
        sampling = result.get("sampling_seconds")
        predict = result.get("prediction_seconds")
        acceptance = result.get("acceptance_fraction")
        divergences = result.get("n_divergent")
        posterior = result.get("posterior_path", "")
        lines.append(
            "| {engine} | {sampler} | {status} | {sampling} | {predict} | {acceptance} | {divergences} | `{posterior}` | {note} |".format(
                engine=result["engine"],
                sampler=result.get("sampler", ""),
                status=result["status"],
                sampling="" if sampling is None else f"{sampling:.3g}",
                predict="" if predict is None else f"{predict:.3g}",
                acceptance="" if acceptance is None else f"{acceptance:.3g}",
                divergences="" if divergences is None else str(divergences),
                posterior=Path(posterior).name if posterior else "",
                note=result.get("note", "").replace("|", "\\|"),
            )
        )
    lines.append("")
    lines.append("## Posterior Medians")
    lines.append("")
    for result in summary["results"]:
        if result["status"] != "ok":
            continue
        lines.append(f"### {result['engine']}")
        lines.append("")
        lines.append("| Parameter | q16 | q50 | q84 |")
        lines.append("|---|---:|---:|---:|")
        for name, q in result["quantiles"].items():
            lines.append(f"| `{name}` | {q['q16']:.5g} | {q['q50']:.5g} | {q['q84']:.5g} |")
        lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append(summary["command"])
    lines.append("```")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anisotropy", choices=("isotropic", "om"), default="isotropic")
    parser.add_argument(
        "--engines",
        default="jeanspy,jeanspy-nuts,jampy,agama,gravsphere-v1,gravsphere2,pygravsphere",
    )
    parser.add_argument("--mock-source", choices=("auto", "create_mock", "jeanspy"), default="auto")
    parser.add_argument("--n-stars", type=int, default=4000)
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--velocity-error-kms", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-walkers", type=int, default=18)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--n-burn", type=int, default=30)
    parser.add_argument("--nuts-warmup", type=int, default=80)
    parser.add_argument("--nuts-samples", type=int, default=100)
    parser.add_argument("--nuts-chains", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Use a small smoke-test sampler.")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--slow-eval-threshold", type=float, default=0.05)
    parser.add_argument("--jeanspy-n-u", type=int, default=160)
    parser.add_argument("--jeanspy-u-max", type=float, default=2500.0)
    parser.add_argument("--jampy-tracer-ngauss", type=int, default=16)
    parser.add_argument("--jampy-nfw-ngauss", type=int, default=24)
    parser.add_argument("--jampy-epsrel", type=float, default=3e-3)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.quick:
        args.n_walkers = 12 if args.anisotropy == "isotropic" else 14
        args.n_steps = 40
        args.n_burn = 15
        args.nuts_warmup = 20
        args.nuts_samples = 30
        args.nuts_chains = 1
        args.jeanspy_n_u = min(args.jeanspy_n_u, 96)
    args.output_dir = args.output_dir.resolve()
    args.report_path = args.report_path.resolve()
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mock = generate_mock(
        output_dir=args.output_dir,
        anisotropy=args.anisotropy,
        n_stars=args.n_stars,
        seed=args.seed,
        velocity_error_kms=args.velocity_error_kms,
        true_params=TRUE_PARAMS,
        source=args.mock_source,
    )
    profile = make_dispersion_profile(mock["R_pc"], mock["vlos_kms"], n_bins=args.n_bins)
    profile_path = args.output_dir / f"profile_{args.anisotropy}_n{args.n_stars}_seed{args.seed}.npz"
    np.savez_compressed(
        profile_path,
        radius_pc=profile.radius_pc,
        sigma_kms=profile.sigma_kms,
        sigma_err_kms=profile.sigma_err_kms,
        counts=profile.counts,
    )

    requested_engines = [item.strip().lower() for item in args.engines.split(",") if item.strip()]
    results: list[dict[str, Any]] = []
    for engine_name in requested_engines:
        factory = ENGINE_FACTORIES.get(engine_name)
        if factory is None:
            results.append({"engine": engine_name, "status": "skipped", "note": "unknown engine"})
            continue
        engine = factory(args.anisotropy, profile.radius_pc, args)
        if engine.status != "ok" or engine.predict_sigma is None:
            results.append(
                {
                    "engine": engine.name,
                    "status": engine.status,
                    "note": engine.note,
                    "setup_seconds": engine.setup_seconds,
                }
            )
            continue
        print(f"[run] {engine.name}: setup={engine.setup_seconds:.3g}s")
        results.append(
            run_engine_sampler(engine=engine, anisotropy=args.anisotropy, profile=profile, args=args)
        )

    command = "uv run python scripts/benchmark_jeans_codes.py " + " ".join(sys.argv[1:])
    summary = {
        "anisotropy": args.anisotropy,
        "n_stars": args.n_stars,
        "n_bins_requested": args.n_bins,
        "n_bins_effective": int(profile.radius_pc.size),
        "velocity_error_kms": args.velocity_error_kms,
        "seed": args.seed,
        "true_params": TRUE_PARAMS,
        "mock": {
            "path": mock["path"],
            "source": mock["source"],
            "config_path": mock.get("config_path"),
            "snapshot_path": mock.get("snapshot_path"),
        },
        "profile_path": profile_path,
        "sampler": {
            "n_walkers": args.n_walkers,
            "n_steps": args.n_steps,
            "n_burn": args.n_burn,
            "nuts_warmup": args.nuts_warmup,
            "nuts_samples": args.nuts_samples,
            "nuts_chains": args.nuts_chains,
        },
        "results": results,
        "command": command,
    }
    summary_path = args.output_dir / f"summary_{args.anisotropy}_n{args.n_stars}_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    write_report(summary, args.report_path)
    print(f"[done] summary: {summary_path}")
    print(f"[done] report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
