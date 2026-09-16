"""Keep the downloadable learning path executable and its saved outputs current."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("quickstart", "tutorials/units", "tutorials/backends", "tutorials/models",
             "tutorials/predictions", "tutorials/inference", "tutorials/storage")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_documentation_notebook_has_current_successful_outputs(name):
    notebook = json.loads((ROOT / "docs/source" / f"{name}.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    sources = ["".join(cell["source"]) for cell in cells]
    for index, (cell, code) in enumerate(zip(cells, sources)):
        compile(code, f"{name}:{index}", "exec")
        assert cell["execution_count"] is not None
        assert not any(output["output_type"] == "error" for output in cell["outputs"])
        assert "from docs_" not in code, "Downloads must not import repository example files"
    actual = hashlib.sha256(json.dumps(sources).encode()).hexdigest()
    assert notebook["metadata"]["jeanspy"]["execution"]["code_sha256"] == actual
    if name in {"quickstart", "tutorials/predictions", "tutorials/inference"}:
        assert any("image/png" in output.get("data", {})
                   for cell in cells for output in cell["outputs"])
