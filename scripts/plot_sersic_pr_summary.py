#!/usr/bin/env python3
"""Create a compact visual summary of the Sérsic deprojection implementation.

This script intentionally imports the public API used by downstream users:
``from jeanspy.model import SersicModel``.  Numerical Abel inversion is plotted
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

    # Include points just outside the fast-approximation domain so the plot
    # also demonstrates that auto falls back to numerical Abel inversion.
    x = np.logspace(-5, 3.2, 90)
    r_pc = x * RE_PC

    fig, (ax_density, ax_error) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for n in N_VALUES:
        model = SersicModel(re_pc=RE_PC, n=n)  # default = auto
        rho_num = model.density_3d(r_pc, method="numerical")
        rho_auto = model.density_3d(r_pc)

        # Dimensionless density makes curves independent of the arbitrary Re.
        line, = ax_density.loglog(
            x,
            rho_auto * RE_PC**3,
            lw=1.8,
            label=fr"$n={n:g}$ auto",
        )
        ax_density.loglog(
            x,
            rho_num * RE_PC**3,
            ls="--",
            lw=1.0,
            color=line.get_color(),
            alpha=0.75,
            label=fr"$n={n:g}$ numerical",
        )

        valid = (rho_num > 0.0) & np.isfinite(rho_num) & np.isfinite(rho_auto)
        rel = np.full_like(rho_num, np.nan, dtype=float)
        rel[valid] = np.abs(rho_auto[valid] / rho_num[valid] - 1.0)
        # Zero error occurs where auto deliberately falls back to numerical;
        # plot a tiny floor so it remains visible on a log axis.
        rel_plot = np.where(np.isfinite(rel), np.maximum(rel, 1e-12), np.nan)
        ax_error.loglog(x, rel_plot, lw=1.8, label=fr"$n={n:g}$")

    for ax in (ax_density, ax_error):
        ax.axvline(1e-4, ls=":", lw=1.0, color="0.45")
        ax.axvline(1e3, ls=":", lw=1.0, color="0.45")
        ax.grid(True, which="both", alpha=0.2)
        ax.set_xlabel(r"$r/R_e$")

    ax_density.set_ylabel(r"$\rho(r) R_e^3$")
    ax_density.set_title("Implemented Sérsic 3-D density")
    ax_density.legend(fontsize=7, ncol=2)

    ax_error.axhline(1e-2, ls="--", lw=1.0, color="0.35", label="1%")
    ax_error.axhline(1e-3, ls="-.", lw=1.0, color="0.35", label="0.1%")
    ax_error.set_ylabel(r"$|\rho_{\rm auto}/\rho_{\rm numerical}-1|$")
    ax_error.set_title("Default auto policy vs numerical Abel")
    ax_error.set_ylim(1e-12, 2e-1)
    ax_error.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "JeansPy PR #24: VM20bis (n≤3.4) / SP04 (n>3.4) with numerical fallback",
        fontsize=11,
    )
    fig.tight_layout()

    png = output.with_suffix(".png")
    svg = output.with_suffix(".svg")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(png)
    print(svg)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sersic_deprojection_summary")
