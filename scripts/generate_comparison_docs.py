"""Render the source-backed software matrix from its inspectable JSON records."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "validation/release/comparison.json").read_text())
    rows = data["records"]
    content = ["# Related software and comparison questions", "",
               "Source review date: **" + data["inspected_date"] + "**. " + data["interpretation"], "",
               "This survey has not established a speed or inference-quality ranking. "
               "Published performance examples use different workloads and hardware; "
               "they are not substituted for measurements in this project.", "",
               "The working hypothesis is that gradients of direct spherical and "
               "axisymmetric Jeans models can help inference while retaining numerical "
               "accuracy. SKiNN already provides differentiable emulation, GLaD already "
               "uses JAX/GPU Jeans calculations, and JamPy 9 introduces a substantially "
               "different spectral algorithm. None permits a novelty claim based on "
               "differentiability or GPU support alone. CJAM and later discrete JAM "
               "studies also establish individual stellar velocities and contamination "
               "models as existing methods.", "",
               "## Physical models and observations", ""]
    groups = [
        [("code", "Code"), ("target", "Target"), ("geometry", "Geometry"),
         ("physics", "Physical assumptions"), ("observables", "Observables / likelihood"),
         ("freedom", "Model freedom")],
        [("code", "Code"), ("solver", "Numerical method"),
         ("differentiation", "Differentiation"), ("hardware", "CPU / GPU"),
         ("inference", "Inference")],
        [("code", "Code"), ("public_status", "Public availability"),
         ("code_version", "Code surveyed / measured"), ("paper_version", "Paper")],
    ]
    for index, fields in enumerate(groups):
        if index:
            content += ["## " + ["", "Numerics and inference", "Versions and availability"][index], ""]
        content += ["| " + " | ".join(title for _, title in fields) + " |",
                    "| " + " | ".join("---" for _ in fields) + " |"]
        for row in rows:
            vals = [row[key].replace("|", "\\|") for key, _ in fields]
            vals[0] += " [sources](#" + anchor(row["code"]) + ")"
            content.append("| " + " | ".join(vals) + " |")
        content.append("")
    content += ["## Comparison boundaries and sources", ""]
    for row in rows:
        content += ["(" + anchor(row["code"]) + ")=", "### " + row["code"], "",
                    row["comparison_scope"], "",
                    "; ".join(f"[{label}]({url})" for label, url in row["sources"]) + ".", ""]
    content += ["The [current JAM validation](../validation/jam9.md) records the selected",
                "isotropic accuracy tests, their failed coarse setting and their limits.", "",
                "## Claims requiring new measurements", "",
                "1. Accuracy and physical-parameter gradients over a declared parameter domain.",
                "2. Prediction and gradient cost including preparation, compilation and transfers.",
                "3. Agreement of posterior summaries under identical priors and likelihoods.",
                "4. Time to a predeclared inference-precision target, counting failed runs.",
                "5. Mock recovery and interval coverage with simulation uncertainty.", "",
                "Full-model calibration, general-Hessian accuracy, membership mixtures and "
                "variational inference are not claimed merely because JAX or NumPyro can "
                "represent them. Current NumPy J/D integration remains separate from "
                "the differentiated kinematic likelihood.", "",
                "The machine-readable source is "
                "{download}`comparison.json <../../../validation/release/comparison.json>`. "
                "Regenerate this page with `python scripts/generate_comparison_docs.py`.", ""]
    (ROOT / "docs/source/comparison/index.md").write_text("\n".join(content))


def anchor(code):
    import re
    return "software-" + re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-")


if __name__ == "__main__":
    main()
