#!/usr/bin/env python3
"""Render the completed frozen LOS benchmark without executing a model."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"validation/release/los-benchmark"
FIGURES = ROOT/"docs/source/_static/validation"
COLORS = {"numpy": "#24649c", "jax-cpu": "#b87916", "jax-gpu": "#67753b", "jam": "#a2497a"}
LABELS = {"numpy": "JeansPy NumPy", "jax-cpu": "JeansPy JAX CPU", "jax-gpu": "JeansPy JAX GPU", "jam": "JAM 9.0.2 CPU"}
MARKERS = {"numpy": "o", "jax-cpu": "^", "jax-gpu": "v", "jam": "D"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "svg.hashsalt": "jeanspy-matched-los-v1", "pdf.fonttype": 42})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return "unavailable" if value is None else f"{value:.4g}"


def main():
    protocol_path = ROOT/"validation/release/los_benchmark_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    files = [protocol_path, ROOT/"scripts/benchmark_release_los.py"]
    hardware = json.loads((DATA/"hardware.json").read_text())
    files += [DATA/"hardware.json", DATA/"budget_accounting.json"]
    reference = json.loads((DATA/"reference.json").read_text())
    if reference["status"] != "completed":
        raise ValueError("the independent reference must pass before interpretation")
    files.append(DATA/"reference.json")
    rows, reports, memory = [], {}, []
    for role in COLORS:
        path = DATA/(role+".json")
        report = json.loads(path.read_text())
        supervisor_path = DATA/(role+"-supervisor.json")
        supervisor = json.loads(supervisor_path.read_text())
        files += [path, supervisor_path, DATA/(role+"-stdout.txt")]
        if report["status"] not in ("completed", "completed_with_execution_failures"):
            raise ValueError(f"unfinished role: {role}")
        if report["protocol_sha256"] != sha(protocol_path) or report["reference_sha256"] != sha(DATA/"reference.json"):
            raise ValueError(f"source identity mismatch: {role}")
        if report["script_sha256"] != sha(ROOT/"scripts/benchmark_release_los.py"):
            raise ValueError(f"the executed runner differs from the published runner: {role}")
        expected = 18 if role == "jam" else 27
        if len(report["rows"]) != expected:
            raise ValueError(f"missing benchmark rows: {role}")
        reports[role] = report
        memory.append(dict(role=role, host_gib=supervisor["peak_host_rss_bytes"]/2**30,
            device_gib=supervisor["peak_gpu_bytes"]/2**30, elapsed_seconds=supervisor["elapsed_seconds"]))
        for source in report["rows"]:
            errors = [source.get("accuracy", {})] + [v["accuracy"] for v in source.get("repetitions", [])]
            finite = all(e.get("finite_positive", False) for e in errors)
            row = dict(role=role, solver=source["solver"], q=source["q"], n_positions=source["n_positions"],
                tier=source["tier"], setting=json.dumps(source["setting"], sort_keys=True), status=source["status"],
                max_relative_second_moment=max(e["max_relative_second_moment"] for e in errors) if finite else None,
                max_relative_sigma=max(e["max_relative_sigma"] for e in errors) if finite else None,
                first_ms=1000*source["first_prediction_seconds"] if "first_prediction_seconds" in source else None,
                warm_ms=1000*source["warm"]["median_seconds"] if "warm" in source else None,
                p10_ms=1000*source["warm"]["p10_seconds"] if "warm" in source else None,
                p90_ms=1000*source["warm"]["p90_seconds"] if "warm" in source else None,
                passed=source.get("passed", False))
            if row["warm_ms"] is not None and len(source["repetitions"]) != protocol["timing"]["warm_repeats"]:
                raise ValueError("an incomplete warm series cannot enter a timing plot")
            rows.append(row)
    out = DATA/"summary.csv"
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    outputs = [out]
    out = DATA/"summary.json"
    out.write_text(json.dumps(dict(rows=rows, memory=memory), indent=2)+"\n")
    outputs.append(out)
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8), sharex=True, sharey=True)
    for axrow, q in zip(axes, protocol["physical"]["tracer_q"]):
        for ax, count in zip(axrow, protocol["coordinates"]["counts"]):
            for role, color in COLORS.items():
                solvers = ["jam"] if role == "jam" else ["axisymmetric", "spherical"]
                for solver in solvers:
                    selected = [r for r in rows if r["role"] == role and r["solver"] == solver
                        and r["q"] == q and r["n_positions"] == count and r["warm_ms"] is not None
                        and r["max_relative_second_moment"] is not None]
                    if not selected:
                        continue
                    x = np.array([r["warm_ms"] for r in selected])
                    y = np.maximum([r["max_relative_second_moment"] for r in selected], 1e-14)
                    error = np.array([[r["warm_ms"]-r["p10_ms"] for r in selected],
                        [r["p90_ms"]-r["warm_ms"] for r in selected]])
                    marker = MARKERS[role]
                    ax.errorbar(x, y, xerr=error, color=color, marker=marker, ms=5,
                        markerfacecolor="white" if solver == "spherical" else color,
                        linestyle="--" if solver == "spherical" else "-", linewidth=.9, capsize=2)
            ax.axhline(.005, color="#555555", linestyle=":", linewidth=1)
            ax.set(xscale="log", yscale="log", ylim=(5e-15, .1), title=f"q = {q:g}, {count} positions")
            ax.grid(axis="y", color="#e8e8e8", linewidth=.5)
    for ax in axes[-1]:
        ax.set_xlabel("Warm evaluation time [ms]")
    for ax in axes[:, 0]:
        ax.set_ylabel("LOS second-moment error\n(maximum relative)")
    handles = [Line2D([], [], color=color, marker=MARKERS[role], lw=1, label=LABELS[role]) for role, color in COLORS.items()]
    handles += [Line2D([], [], color="#555555", marker="o", label="Filled: axisymmetric", lw=0),
        Line2D([], [], color="#555555", marker="o", mfc="white", label="Open: spherical specialization", lw=0),
        Line2D([], [], color="#555555", ls=":", label="0.5% accuracy threshold")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .95), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Matched Plummer LOS accuracy and evaluation time", y=.995, fontsize=14)
    fig.text(.5, .018, "Three frozen settings per curve; horizontal bars: p10–p90 of ten calls. Errors below 10⁻¹⁴ are placed at 10⁻¹⁴.",
        ha="center", fontsize=9)
    fig.subplots_adjust(top=.80, bottom=.105, left=.085, right=.985, hspace=.25, wspace=.13)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png", "pdf"):
        path = FIGURES/("los_accuracy_time."+suffix)
        metadata = {"Date": None} if suffix == "svg" else ({"CreationDate": None, "ModDate": None} if suffix == "pdf" else {})
        fig.savefig(path, dpi=180, metadata=metadata)
        if suffix == "svg":
            path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines())+"\n")
        outputs.append(path)
    plt.close(fig)
    passed = sum(r["passed"] for r in rows)
    failed = [r for r in rows if not r["passed"]]
    lines = ["# LOS dispersion: accuracy and time", "",
        "This forward-only benchmark compares JeansPy's NumPy and JAX solvers with JAM 9.0.2 on matched Plummer models. MCMC comparison is outside the release scope by the author's 2026-09-14 decision. The protocol and runner were fixed in commit `b3f1545` before any new evaluation. The physical cases and JAM settings were already studied in the earlier [JAM validation](jam9.md); this is a fresh timing experiment, not an unseen-model test.", "",
        "## Physical comparison", "",
        "The potential is spherical Plummer with mass $10^7 M_\\odot$ and scale 300 pc. The tracer has the same scale, intrinsic axis ratio $q=1$ or $0.7$, inclination 1.1 rad, and zero meridional anisotropy. Coordinates follow the same projected ellipses from 10 to 1500 pc. The observable is $\\overline{v_{\\rm los}^2}$ in $(\\mathrm{km\\,s^{-1}})^2$. Its square root is $\\sigma_{\\rm los}$ for the imposed zero-mean-velocity convention, and $v_{\\rm rms}$ otherwise. There is no PSF, pixel averaging, measurement error or fitted streaming model.", "",
        "JAM's spherical alignment and JeansPy's cylindrical alignment give the same even-moment equations in this zero-anisotropy subset. JAM takes an analytic potential containing the same gravitational constant, plus a fitted MGE tracer. The MGE approximation is part of its reported total prediction error. Its fitting time is reported separately and is reusable while the tracer is fixed. The specialized spherical JeansPy solver is included only for $q=1$. The general axisymmetric solver remains present in that case for a like-geometry comparison. See [Cappellari (2026)](https://arxiv.org/abs/2601.16179) and the [pinned JAM documentation](https://pypi.org/project/jampy/9.0.2/).", "",
        "## Accuracy versus time", "",
        "```{figure} ../_static/validation/los_accuracy_time.svg", ":alt: Six panels compare LOS second-moment error against synchronized warm time for two tracer shapes, three position counts, three numerical settings and four CPU or GPU implementations.", ":width: 100%", "",
        "Each marker is a measured setting; connecting segments only identify the three fixed configurations. Error is the largest relative discrepancy over all positions and all eleven mass normalizations. Horizontal bars show the 10th–90th percentiles of ten warm calls, not uncertainty across independent machine sessions. GPU timings synchronize completion and exclude separately recorded transfers. The dotted line is the predeclared 0.5% second-moment criterion. Errors below $10^{-14}$ are plotted at that floor, with exact values retained in the tables and raw data.", "```", "",
        f"All {len(rows)} declared rows were attempted; {passed} pass the 0.5% criterion and {len(failed)} do not. A completed timing row can fail numerical accuracy. The full table below retains every setting; no fastest-setting selection or universal speedup is inferred.", "",
        "All first-call times include the ordinary forward evaluation; JAX also includes tracing and compilation. Warm calls use the same ten prescribed mass-normalization changes (within one percent), with input shapes and tracer fixed. JAX physical parameters and coordinates are dynamic arguments. NumPy and JAM recompute their public forward call. No output is substituted by rescaling an earlier prediction.", "",
        "## Complete measured settings", "",
        "Times are milliseconds. The maximum error includes the first call and every warm prediction. The square-root error is computed directly from the same moment ratio. JAM settings list MGE component count, angular order and LOS order; JeansPy settings give its fixed quadrature order. Spherical mass quadrature has 128 nodes in both JeansPy implementations.", "",
        "| Engine | Solver | q | Positions | Setting | First [ms] | Warm median [ms] | Moment error | Sigma error | 0.5% gate |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        setting = r["setting"].replace('{"nang": ', "").replace(', "ngauss": ', "/").replace(', "nlos": ', "/").replace("}", "") if r["role"] == "jam" else r["setting"]
        if r["role"] == "jam":
            cfg = json.loads(r["setting"])
            setting = f"{cfg['ngauss']}/{cfg['nang']}/{cfg['nlos']}"
        lines.append(f"| {LABELS[r['role']]} | {r['solver']} | {r['q']:g} | {r['n_positions']} | {setting} | {number(r['first_ms'])} | {number(r['warm_ms'])} | {number(r['max_relative_second_moment'])} | {number(r['max_relative_sigma'])} | {'pass' if r['passed'] else 'fail'} |")
    lines += ["", "## MGE setup and memory", "",
        "| Tracer q | Requested Gaussian components | Actual components | Fit time [s] | Maximum density error, 1–10,000 pc |",
        "| ---: | ---: | ---: | ---: | ---: |"]
    for fit in reports["jam"]["mge_fits"]:
        lines.append(f"| {fit['q']:g} | {fit['requested_components']} | {fit['actual_components']} | {fit['seconds']:.4g} | {fit['density_relative_error']:.4g} |")
    lines += ["", "Each memory entry is the supervisor-observed process high-water mark across the entire role, including model construction, all sizes/settings and JAX compilation. GPU allocations include runtime reservations; these are not incremental per-kernel memory requirements.", "",
        "| Role | Peak host RSS [GiB] | Observed GPU allocation [GiB] | Role elapsed time [s] |", "| --- | ---: | ---: | ---: |"]
    for m in memory:
        lines.append(f"| {LABELS[m['role']]} | {m['host_gib']:.3f} | {m['device_gib']:.3f} | {m['elapsed_seconds']:.3f} |")
    lines += ["", "## Independent reference and limits", "",
        r"For $q=1$, the reference is the exact expression $3\pi GM/(64\sqrt{a^2+x^2+y^2})$. For $q=0.7$, use dimensionless $a=GM=1$, $A=1+R^2$, $T=A+z^2$, $C=3/(4\pi q)$ and $w=(1-q^2)A/T$. Direct integration of the vertical Jeans equation gives", "",
        "```{math}", r"p\equiv\nu\sigma_z^2=\frac{Cq^5}{6T^3}\,{}_2F_1(5/2,3;4;w).", "```", "",
        r"Its analytic radial derivative, together with $\nu\overline{v_\phi^2}=p+R\partial_Rp+R\nu\partial_R\Phi$, gives the LOS integrand $p+\sin^2(i)x^2[(\partial_Rp)/R+\nu/T^{3/2}]$. SciPy adaptive quadrature integrates over the infinite LOS, then divides by the exact projected tracer density and restores the factor $GM/a$. This calculation uses no JeansPy force or numerical-integration helper. Pressure and radial derivative are checked against separate direct integrals at eight intrinsic points; both projection tolerances are evaluated at every sky position. The spherical numerical projection is additionally checked against the exact formula. All reference checks pass the fixed $10^{-8}$ relative criteria; the quadrature error estimates and every prediction remain in the reference record.", "",
        "The experiment covers two smooth models with spherical gravity, zero meridional anisotropy, one host and one GPU. It does not establish accuracy for cusps, flattened halos, finite cutoffs, nonzero anisotropies or a whole prior domain. Higher quadrature order has a measured cost; low-order agreement here does not justify a default change. No runtime scientific definition or numerical default was changed.", "",
        "## Reproduction and source data", "",
        f"CPU: {reports['numpy']['cpu_model']}, logical CPUs 0–7, BLAS/OpenMP threads set to one. JAX CPU/GPU use float64. GPU: {hardware['gpu_name']} ({hardware['total_memory_mib']/1024:g} GiB, driver {hardware['driver_version']}); process device {', '.join(reports['jax-gpu']['devices'])}. Dependency versions, executable paths, CPU affinity, protocol/runner/runtime hashes, every repetition, warnings and observed memory are retained in the JSON and supervisor files.", "",
        "Download the {download}`complete timing table <../../../validation/release/los-benchmark/summary.csv>`, {download}`frozen protocol <../../../validation/release/los_benchmark_protocol.json>` and {download}`independent reference <../../../validation/release/los-benchmark/reference.json>`. The per-role raw outputs are {download}`NumPy <../../../validation/release/los-benchmark/numpy.json>`, {download}`JAX CPU <../../../validation/release/los-benchmark/jax-cpu.json>`, {download}`JAX GPU <../../../validation/release/los-benchmark/jax-gpu.json>` and {download}`JAM <../../../validation/release/los-benchmark/jam.json>`.", "",
        "Regenerate this page and its figures without numerical experiments:", "", "```bash", "python scripts/render_release_los.py", "```", "",
        "For a new experiment, use the committed protocol with `scripts/benchmark_release_los.py --role reference --output <new-reference.json>`, followed by roles `numpy`, `jax-cpu`, `jax-gpu` and `jam`, each with `--reference <new-reference.json>` and a fresh output path. Launch through the cumulative-budget supervisor with the role's frozen time/memory bound, `taskset -c 0-7`, `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. Configure `JEANSPY_JAX_PLATFORM=cpu` or `gpu` and `JEANSPY_JAX_ENABLE_X64=true` before the corresponding Python process starts. JAM uses a separate environment with `jampy==9.0.2` and `mgefit==6.2.6`. Exact commands and dependency versions for these runs are in the retained supervisor/role records. The current campaign uses one existing 24-hour ledger; this page does not authorize resetting or extending it.", ""]
    page = ROOT/"docs/source/validation/los-benchmark.md"
    page.write_text("\n".join(lines))
    outputs.append(page)
    manifest = dict(renderer_sha256=sha(Path(__file__)),
        inputs=[dict(path=str(path.relative_to(ROOT)), sha256=sha(path)) for path in files],
        outputs=[dict(path=str(path.relative_to(ROOT)), sha256=sha(path)) for path in outputs])
    (DATA/"figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(rows=len(rows), passed=passed, failed=len(failed))))


if __name__ == "__main__":
    main()
