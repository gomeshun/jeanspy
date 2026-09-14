"""Execute the exact short Python files included in the documentation."""
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("example", ["docs_spherical.py", "docs_axisymmetric.py", "docs_jax.py",
    "docs_profiles.py", "docs_factors.py", "docs_numerics.py", "docs_jax_spherical.py",
    "docs_inference.py", "docs_numpyro_inference.py"])
def test_documented_example(example):
    if example in {"docs_jax.py", "docs_jax_spherical.py", "docs_numerics.py", "docs_numpyro_inference.py"}:
        pytest.importorskip("jax")
    if example == "docs_numpyro_inference.py":
        pytest.importorskip("numpyro")
        pytest.importorskip("h5netcdf")
    runpy.run_path(str(ROOT / "examples" / example), run_name="__main__")
