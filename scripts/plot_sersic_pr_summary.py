#!/usr/bin/env python3
"""Create a compact visual summary of the Sérsic deprojection implementation.

This script intentionally imports the public API used by downstream users:
``from jeanspy.model import SersicModel``. Numerical Abel inversion is plotted
as the reference; the default ``auto`` policy is overlaid and its relative
error is shown for representative dwarf-galaxy Sérsic indices.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

from jeanspy.model import SersicModel


RE_PC = 100.0
N_VALUES = (0.6, 1.0, 2.0, 4.0)


def main(output_stem: str = "sersic_deprojection_summary") -> None:
    output = pathlib.Path(output_stem)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Keep the density panel in the radial range where all representative
    # profiles remain visually informative. The error panel extends farther
    # out to show approximation quality across a broad Jeans-relevant range.
    x_density = np.logspace(-4, np.log10(5.0), 90)
    x_error = np.logspace(-4, 1.0, 100)

    fig, (ax_density, ax_error) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    print("auto-vs-numerical relative-error summary for 1e-4 <= r/Re <= 10")
    print("n      method      max_rel_err      rms_rel_err")

    for n in N_VALUES:
        model = SersicModel(re_pc=RE_PC, n=n)  # default = auto

        r_density = x_density * RE_PC
        rho_num_density = model.density_3d(r_density, method="numerical")
        rho_auto_density = model.density_3d(r_density)

        # Dimensionless density makes curves independent of the arbitrary Re.
        line, = ax_density.loglog(
            x_density,
            rho_auto_density * RE_PC**3,
            lw=2.0,
            label=fr"$n={n:g}$ auto",
        )
        ax_density.loglog(
            x_density,
            rho_num_density * RE_PC**3,
            ls="--",
            lw=1.1,
            color=line.get_color(),
            alpha=0.8,
            label=fr"$n={n:g}$ numerical",
        )

        r_error = x_error * RE_PC
        rho_num = model.density_3d(r_error, method="numerical")
        rho_auto = model.density_3d(r_error)
        valid = (rho_num > 0.0) & np.isfinite(rho_num) & np.isfinite(rho_auto)
        rel = np.full_like(rho_num, np.nan, dtype=float)
        rel[valid] = np.abs(rho_auto[valid] / rho_num[valid] - 1.0)
        rel_plot = np.where(np.isfinite(rel), np.maximum(rel, 1e-12), np.nan)
        ax_error.loglog(x_error, rel_plot, lw=2.0, label=fr"$n={n:g}$")

        finite = rel[np.isfinite(rel)]
        method = "VM20bis" if n <= 3.4 else "SP04"
        print(
            f"{n:<4g}   {method:<9s}   {np.max(finite):.6e}   "
            f"{np.sqrt(np.mean(finite**2)):.6e}"
        )

    for ax in (ax_density, ax_error):
        ax.grid(True, which="both", alpha=0.2)
        ax.set_xlabel(r"$r/R_e$")

    ax_density.set_ylabel(r"$\rho(r) R_e^3$")
    ax_density.set_title("Implemented Sérsic 3-D density")
    ax_density.set_xlim(1e-4, 5.0)
    ax_density.set_ylim(1e-9, 10.0)
    ax_density.legend(fontsize=7, ncol=2)

    ax_error.axhline(1e-2, ls="--", lw=1.0, color="0.35", label="1%")
    ax_error.axhline(1e-3, ls="-.", lw=1.0, color="0.35", label="0.1%")
    ax_error.set_ylabel(r"$|\rho_{\rm auto}/\rho_{\rm numerical}-1|$")
    ax_error.set_title("Default auto policy vs numerical Abel")
    ax_error.set_xlim(1e-4, 10.0)
    ax_error.set_ylim(1e-6, 5e-2)
    ax_error.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "JeansPy PR #24: VM20bis (n≤3.4) / SP04 (n>3.4)",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.005,
        "auto uses numerical Abel outside the calibrated approximation domain "
        "(1e-4 ≤ r/Re ≤ 1e3 or 0.5 ≤ n ≤ 10).",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))

    png = output.with_suffix(".png")
    svg = output.with_suffix(".svg")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(png)
    print(svg)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sersic_deprojection_summary")
