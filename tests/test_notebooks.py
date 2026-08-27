import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
PUBLIC_NOTEBOOKS = {
    "demo_model_full.ipynb",
    "sampler_numpyro_demo.ipynb",
    "benchmark_jeans_codes.ipynb",
}
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users|tmp|workspaces|mnt)/)")


def _strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_public_notebook_inventory_is_explicit():
    assert {path.name for path in NOTEBOOKS.glob("*.ipynb")} == PUBLIC_NOTEBOOKS


def test_public_notebooks_are_output_free_and_portable():
    for name in PUBLIC_NOTEBOOKS:
        path = NOTEBOOKS / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                assert cell["execution_count"] is None
        assert not any(LOCAL_PATH.search(value) for value in _strings(notebook))


def test_public_notebook_code_cells_compile():
    for name in PUBLIC_NOTEBOOKS:
        path = NOTEBOOKS / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, f"{path}:{index}", "exec")


def test_generated_notebook_artifacts_are_ignored():
    generated = (
        "notebooks/_demo_outputs/sampler_numpyro_demo/last_state.pkl",
        "notebooks/example.nc",
        "notebooks/example.zarr/data",
        "notebooks/example.pkl",
    )
    for path in generated:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, f"{path} should be ignored"


def test_readme_links_retained_notebooks():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in PUBLIC_NOTEBOOKS:
        assert f"notebooks/{name}" in readme
