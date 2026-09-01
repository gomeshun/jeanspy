import json
from pathlib import Path

import matplotlib


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "demo_model_full.ipynb"


def test_canonical_getting_started_notebook_executes():
    matplotlib.use("Agg", force=True)

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__jeanspy_notebook_test__"}

    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            exec(
                compile(source, f"{NOTEBOOK}#cell-{index}", "exec"),
                namespace,
                namespace,
            )
    finally:
        import matplotlib.pyplot as plt

        plt.close("all")
