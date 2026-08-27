#!/usr/bin/env python3
"""Diagnostic script: compare SersicModel numerical vs LGM deprojection.

Usage
-----
    python scripts/compare_sersic_deprojection.py            # interactive display
    python scripts/compare_sersic_deprojection.py output.png # save to file

Expected qualitative result
----------------------------
* Near r ~ R_e the two methods agree at the percent level for all n.
* For low Sérsic index (n ≲ 1), LGM departs significantly at small r / R_e.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# Allow running from repository root without installing the package.
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jeanspy.model import SersicModel  # noqa: E402


def main(output_path=None):
    ns = [0.56, 0.75, 1.0, 2.0, 4.0, 8.0]
    re_pc = 100.0
    x_arr = np.logspace(-2, 1, 200)  # r / R_e

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab10")

    ax_density, ax_rel = axes

    for i, n in enumerate(ns):
        color = cmap(i)
        m = SersicModel(re_pc=re_pc, n=n)
        r_arr = x_arr * re_pc

        rho_num = m.density_3d_numerical(r_arr)
        rho_lgm = m.density_3d_LGM(r_arr)

        ax_density.loglog(x_arr, rho_num, color=color, lw=1.5, label=f"n={n} (num)")
        ax_density.loglog(x_arr, rho_lgm, color=color, lw=1.5, ls="--")

        rel_err = np.abs(rho_lgm / rho_num - 1.0)
        ax_rel.loglog(x_arr, rel_err, color=color, label=f"n={n}")

    ax_density.set_xlabel(r"$r / R_e$")
    ax_density.set_ylabel(r"$\rho(r)$ [a.u.]")
    ax_density.set_title("3-D Sérsic density: numerical (solid) vs LGM (dashed)")
    ax_density.legend(fontsize=7)

    ax_rel.axhline(0.01, color="gray", ls=":", lw=1, label="1% level")
    ax_rel.set_xlabel(r"$r / R_e$")
    ax_rel.set_ylabel(r"$|\rho_\mathrm{LGM}/\rho_\mathrm{num} - 1|$")
    ax_rel.set_title("Relative error: LGM vs numerical Abel inversion")
    ax_rel.legend(fontsize=7)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    main(out)
