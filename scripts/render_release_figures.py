"""Regenerate release-validation figures and tables from retained JSON results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#2864A0", "#9B6A12", "#272727"]


def render_jam9(output_dir):
    source = ROOT/"validation/release/jam9_plummer_v1.json"
    report = json.loads(source.read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.facecolor": "white",
        "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.hashsalt": "jeanspy-jam9-validation-v1"})
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.0), sharex="col",
                              gridspec_kw={"height_ratios": [1.3, 1.]}, layout="constrained")
    xy = report["protocol"]["sky_coordinates_pc"]
    x, y = np.asarray(xy["x"]), np.asarray(xy["y"])
    for column, case in enumerate(report["cases"]):
        q, inc = case["q"], case["params"]["inclination"]
        qproj = np.sqrt(np.cos(inc)**2+q*q*np.sin(inc)**2)
        radius = np.hypot(x, y/qproj)
        reference = np.asarray(case["jeanspy"][-1]["projected"])
        ax = axes[0, column]
        ax.scatter(radius, reference, s=38, facecolors="none", edgecolors="black",
                   label="JeansPy, 128 nodes", linewidths=1.1)
        ax.scatter(radius, case["jam_projected"][-1]["projected"], s=23, marker="+",
                   color=COLORS[1], label="JAM 9, finest setting", linewidths=1.4)
        ax.set_title(f"Plummer tracer: intrinsic q = {q:g}")
        ax.set_ylabel(r"LOS second moment [(km s$^{-1}$)$^2$]")
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
        ax.legend(fontsize=8, frameon=False)
        residual = axes[1, column]
        for index, (row, marker) in enumerate(zip(case["jam_projected"], ["x", "s", "+"])):
            value = np.asarray(row["projected"])
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite projected values need an explicitly revised figure")
            residual.scatter(radius, 100*(value/reference-1), s=28, marker=marker,
                color=COLORS[index], label=f"{row['ngauss']} MGE / {row['nang']} angular / {row['nlos']} LOS")
        residual.axhline(0., color="#777777", linewidth=.7)
        residual.set_xscale("log")
        residual.set_xlabel(r"Elliptical projected radius [pc]")
        residual.set_ylabel("JAM / JeansPy − 1 [%]")
        residual.grid(axis="y", color="#dddddd", linewidth=.5)
        residual.legend(fontsize=7.2, frameon=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    products = []
    for suffix in ["pdf", "svg", "png"]:
        path = output_dir/f"jam9_validation.{suffix}"
        metadata = {"Creator": "JeansPy release validation"}
        if suffix == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        elif suffix == "svg":
            metadata.update(Date=None)
        else:
            metadata = {"Software": "JeansPy release validation"}
        fig.savefig(path, dpi=180, metadata=metadata)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        products.append(path)
    plt.close(fig)

    lines = ["# Current JAM 9 accuracy comparison", "",
        "The first run of `jam9-plummer-isotropic-v1` used the protocol and runner",
        "fixed in commit `51efc65`, before evaluating the new reference. Its raw",
        "result, source hashes, failed coarse evaluations and follow-up warning",
        "diagnostic are retained under `validation/release/`.", "",
        "The two cases use a spherical Plummer gravitational potential with mass",
        "$10^7\\,M_\\odot$ and scale 300 pc, a normalized Plummer tracer with intrinsic",
        "$q=1$ or $0.7$, and inclination 1.1 rad. Both solvers use zero meridional",
        "anisotropy. In this subset the spherical/cylindrical alignment distinction",
        "vanishes; nonzero anisotropies do not define equivalent closures.", "",
        "JAM receives the analytic potential. Its intrinsic calculation also accepts",
        "an analytic tracer, whereas its projected solver requires a fitted MGE tracer.",
        "The source and domain distinction follows [Cappellari (2026)](https://arxiv.org/abs/2601.16179)",
        "and the [JamPy 9.0.2 documentation](https://pypi.org/project/jampy/9.0.2/).", "",
        "| Intrinsic tracer q | Finest intrinsic difference | Finest projected difference | Projected JAM refinement | Finest MGE density error |",
        "| --- | --- | --- | --- | --- |"]
    for case in report["cases"]:
        e = case["errors"]
        values = [e[k] for k in ["intrinsic_cross_code", "projected_cross_code", "jam_projected_refinement", "mge_density"]]
        lines.append(f"| {case['q']:g} | " + " | ".join(f"{100*v:.5g}%" for v in values) + " |")
    lines += ["", "Differences are maximum absolute relative differences over the six fixed",
        "test positions, in the **second moment**, not its square root. Refinement",
        "compares intermediate and fine configurations. The 0.5% cross-code and JAM",
        "refinement criteria were fixed before execution. The MGE density error is",
        "measured over 1–10,000 pc. It is an approximation diagnostic, not an error bar.", "",
        "```{figure} ../_static/validation/jam9_validation.svg",
        ":alt: Two Plummer tracer cases showing projected second moments and signed JAM residuals at three MGE, angular and LOS resolutions.",
        ":width: 100%", "",
        "Retained projected predictions at six positions. Lower panels show the",
        "combined effect of tracer MGE fitting and projected-solver refinement;",
        "these residuals do not isolate either contribution. Each marker is an actual",
        "evaluation. There are no statistical uncertainty bars in this deterministic test.",
        "```", "", "## Coarse-grid failure and limits", "",
        "JAM's intrinsic 45×9 radial/angular grid returned NaN at every tested",
        "position for both tracer shapes. The 75×15 and 120×21 grids returned finite",
        "results and satisfied the predefined refinement criterion. A bounded",
        "follow-up located the warning in `jam_axi_intr.py`, line 253 of the hashed",
        "9.0.2 source, where the solver takes the logarithm of $r\\,\\partial\\Phi/\\partial r$",
        "for its outer Robin boundary condition. The spherical follow-up reproduced",
        "the coarse failure and found no warning at either finer resolution or with",
        "analytic spatial derivatives at the finest grid. The primary results were",
        "not changed. A diagnostic JSON-writer failure on these NaNs and its repair",
        "are recorded separately.", "",
        "For the spherical case an independent Plummer formula gives",
        "", "```{math}",
        r"\\sigma_r^2(r)=\\frac{GM}{6\\sqrt{a^2+r^2}},\\qquad",
        r"\\overline{v_{\\rm los}^2}(R)=\\frac{3\\pi GM}{64\\sqrt{a^2+R^2}}.",
        "```", "",
        "The stored report includes comparisons against both analytic expressions.",
        "The CI regression additionally compares the NumPy and JAX projected models",
        "with the frozen JAM result and verifies the exact physical scaling",
        "$\\partial\\overline{v_{\\rm los}^2}/\\partial\\rho_s=\\overline{v_{\\rm los}^2}/\\rho_s$.",
        "It does not claim general finite-difference gradient validation from this one derivative.", "",
        "This test covers no flattened gravitational potential, central cusp,",
        "nonzero anisotropy, PSF/pixel averaging, foreground contamination or",
        "posterior coverage. The recorded setup-inclusive times are diagnostics,",
        "not a performance comparison or evidence of a faster solver.", "",
        "## Reproduction", "", "```bash",
        "uv venv /tmp/jeanspy-jam9 --python 3.12",
        'uv pip install --python /tmp/jeanspy-jam9/bin/python ".[plotting]" "jampy==9.0.2" "mgefit==6.2.6"',
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 600s prlimit --as=17179869184 \\",
        "  /tmp/jeanspy-jam9/bin/python scripts/validate_axisymmetric_jam9.py \\",
        "  --output /tmp/jam9-new-result.json",
        "uv run --extra plotting python scripts/render_release_figures.py",
        "```", "",
        "The original environment versions are recorded in the JSON. A different",
        "dependency environment is a new verification, not a byte-identical replay.",
        "JAM's own license applies; its source is not included in the JeansPy distribution.", ""]
    # Math rows use ordinary LaTeX backslashes, not escaped Python text.
    lines = [line.replace("\\\\", "\\") if line.startswith("\\\\") else line for line in lines]
    page = ROOT/"docs/source/validation/jam9.md"
    page.write_text("\n".join(lines))
    manifest = dict(input=str(source.relative_to(ROOT)), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figure_files=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in products])
    (ROOT/"validation/release/figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"docs/source/_static/validation")
    args = parser.parse_args()
    render_jam9(args.output_dir)
