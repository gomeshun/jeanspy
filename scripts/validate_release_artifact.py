#!/usr/bin/env python3
"""Release-gate smoke tests executed against an installed JeansPy artifact."""

from __future__ import annotations

import argparse
import os
import re
from importlib.resources import files
from pathlib import Path

import numpy as np


def _assert_installed_artifact() -> None:
    import jeanspy

    package_path = Path(jeanspy.__file__).resolve()
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        workspace_path = Path(workspace).resolve()
        try:
            package_path.relative_to(workspace_path)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "jeanspy was imported from the source checkout instead of the "
                f"installed artifact: {package_path}"
            )
    print(f"jeanspy imported from {package_path}")


def _extract_quick_start(readme: Path) -> str:
    text = readme.read_text(encoding="utf-8")
    section = re.search(
        r"^## Quick Start\s*$\n(?P<body>.*?)(?=^##\s|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if section is None:
        raise AssertionError("README Quick Start section was not found")

    code_block = re.search(
        r"```python\s*\n(?P<code>.*?)```",
        section.group("body"),
        flags=re.DOTALL,
    )
    if code_block is None:
        raise AssertionError("README Quick Start Python block was not found")
    return code_block.group("code")


def validate_base(readme: Path) -> None:
    _assert_installed_artifact()

    quick_start = _extract_quick_start(readme)
    namespace: dict[str, object] = {"__name__": "__release_quick_start__"}
    exec(compile(quick_start, str(readme), "exec"), namespace, namespace)

    data_dir = files("jeanspy").joinpath("data")
    for filename in ("coeff_dens.csv", "sersic_log10n_log10bn.csv"):
        resource = data_dir.joinpath(filename)
        if not resource.is_file():
            raise AssertionError(f"packaged data file is missing: {filename}")

    from jeanspy.model import SersicModel

    sersic = SersicModel(re_pc=200.0, n=1.0)
    assert sersic.coeff.size > 0
    assert np.isfinite(np.asarray(sersic.coeff, dtype=float)).all()

    print("base Quick Start and packaged-data smoke tests passed")


def validate_numpyro_cpu() -> None:
    _assert_installed_artifact()

    import arviz  # noqa: F401
    import h5netcdf  # noqa: F401
    import jax
    import jax.numpy as jnp
    import netCDF4  # noqa: F401
    import numpyro.distributions as dist
    import xarray  # noqa: F401
    import zarr  # noqa: F401
    from numpyro.handlers import seed, trace

    from jeanspy.model_numpyro import (
        ConstantAnisotropyModel,
        DSphModel,
        NFWModel,
        PlummerModel,
    )
    from jeanspy.sampler_numpyro import JeansLikelihoodModel, ParameterSpec

    if jax.default_backend() != "cpu":
        raise AssertionError(f"expected JAX CPU backend, got {jax.default_backend()!r}")

    dsph = DSphModel(
        submodels={
            "StellarModel": PlummerModel(),
            "DMModel": NFWModel(),
            "AnisotropyModel": ConstantAnisotropyModel(),
        }
    )
    params = {
        "re_pc": 200.0,
        "rs_pc": 1200.0,
        "rhos_Msunpc3": 1.0e-2,
        "r_t_pc": 8000.0,
        "beta_ani": 0.2,
        "vmem_kms": 0.0,
    }
    R_pc = jnp.geomspace(5.0, 200.0, 4)
    e_vlos_kms = jnp.full_like(R_pc, 2.0)
    sigma2 = dsph.sigmalos2(R_pc, params=params, n_u=32, u_max=400.0)
    sigma2_np = np.asarray(jax.block_until_ready(sigma2), dtype=float)
    assert np.isfinite(sigma2_np).all()
    assert (sigma2_np > 0.0).all()

    vlos_kms = params["vmem_kms"] + jnp.sqrt(sigma2 + e_vlos_kms**2) * jnp.array(
        [0.1, -0.2, 0.3, -0.1]
    )
    likelihood = JeansLikelihoodModel(
        dsph,
        [
            ParameterSpec.exp(
                "log_re",
                dist.Normal(jnp.log(200.0), 0.2),
                param_name="re_pc",
            ),
            ParameterSpec.exp(
                "log_rs",
                dist.Normal(jnp.log(1200.0), 0.2),
                param_name="rs_pc",
            ),
            ParameterSpec.exp(
                "log_rhos",
                dist.Normal(jnp.log(1.0e-2), 0.2),
                param_name="rhos_Msunpc3",
            ),
            ParameterSpec.exp(
                "log_r_t",
                dist.Normal(jnp.log(8000.0), 0.2),
                param_name="r_t_pc",
            ),
            ParameterSpec("beta_ani", dist.Uniform(-0.5, 0.8)),
            ParameterSpec("vmem_kms", dist.Normal(0.0, 30.0)),
        ],
        sigmalos2_kwargs={"n_u": 32, "u_max": 400.0},
    )
    model_trace = trace(seed(likelihood, jax.random.PRNGKey(0))).get_trace(
        R_pc=R_pc,
        vlos_kms=vlos_kms,
        e_vlos_kms=e_vlos_kms,
    )
    assert model_trace["re_pc"]["type"] == "deterministic"
    assert model_trace["rs_pc"]["type"] == "deterministic"
    assert model_trace["vlos"]["is_observed"] is True

    print("NumPyro CPU artifact smoke test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", type=Path)
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--numpyro-cpu", action="store_true")
    args = parser.parse_args()

    if args.base == args.numpyro_cpu:
        parser.error("choose exactly one of --base or --numpyro-cpu")
    if args.base:
        if args.readme is None:
            parser.error("--readme is required with --base")
        validate_base(args.readme)
    else:
        validate_numpyro_cpu()


if __name__ == "__main__":
    main()
