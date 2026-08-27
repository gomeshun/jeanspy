#!/usr/bin/env python3
"""Diagnostic script: compare SersicModel deprojection methods.

Compares numerical Abel reference against LGM, VM20, and VM20bis approximations
for representative Sérsic indices, plotting both density profiles and relative
error curves, plus an optional 2D (log n, log r/Re) error map.

Usage
-----
    python scripts/compare_sersic_deprojection.py            # interactive display
    python scripts/compare_sersic_deprojection.py output.png # save to file

The Ciotti, De Deo & Pellegrini (2025, A&A 694 A118) asymptotically matched
approximation is noted as a future benchmark candidate; its formula is not yet
bundled in this repository.

Expected qualitative result
---------------------------
* Near r ~ R_e all analytical methods agree with numerical Abel at the
  percent level for moderate–high n.
* For low n (n ≲ 1), LGM diverges at small r/Re; VM20 and VM20bis
  substantially reduce this error within their respective validity domains.
* VM20bis extends accurate coverage to r/Re ~ 1e-4 for n ≤ 3.4.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings

import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jeanspy.model import SersicModel  # noqa: E402


def _safe_rel_err(rho_approx, rho_ref):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rel = np.abs(rho_approx / rho_ref - 1.0)
    return np.where(np.isfinite(rel), rel, np.nan)


def main(output_path=None):
    re_pc = 100.0

    # ── Representative Sérsic indices ──────────────────────────────────────
    ns = [0.5, 0.56, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0, 10.0]
    cmap = plt.get_cmap("tab10")

    # ── Figure 1: 1-D relative error curves ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    ax_lgm, ax_vm20, ax_vm20bis = axes

    for i, n in enumerate(ns):
        color = cmap(i % 10)
        m = SersicModel(re_pc=re_pc, n=n)

        # LGM domain: n ∈ [0.5, 10]
        if 0.5 <= n <= 10:
            x_lgm = np.logspace(-3, 3, 200)
            rho_lgm = m.density_3d_LGM(np.array(x_lgm) * re_pc)
            rho_num_lgm = m.density_3d_numerical(np.array(x_lgm) * re_pc)
            err_lgm = _safe_rel_err(rho_lgm, rho_num_lgm)
            ax_lgm.loglog(x_lgm, err_lgm, color=color, lw=1.2, label=f"n={n}")

        # VM20 domain: n ∈ [0.5, 10], r/Re ∈ [1e-3, 1e3]
        if 0.5 <= n <= 10:
            x_vm20 = np.logspace(-3, 3, 200)
            rho_vm20 = m.density_3d_VM20(np.array(x_vm20) * re_pc)
            rho_num_vm20 = m.density_3d_numerical(np.array(x_vm20) * re_pc)
            err_vm20 = _safe_rel_err(rho_vm20, rho_num_vm20)
            ax_vm20.loglog(x_vm20, err_vm20, color=color, lw=1.2, label=f"n={n}")

        # VM20bis domain: n ∈ [0.5, 3.4], r/Re ∈ [1e-4, 1e3]
        if 0.5 <= n <= 3.4:
            x_vm20bis = np.logspace(-4, 3, 200)
            rho_vm20bis = m.density_3d_VM20bis(np.array(x_vm20bis) * re_pc)
            rho_num_vm20bis = m.density_3d_numerical(np.array(x_vm20bis) * re_pc)
            err_vm20bis = _safe_rel_err(rho_vm20bis, rho_num_vm20bis)
            ax_vm20bis.loglog(x_vm20bis, err_vm20bis, color=color, lw=1.2, label=f"n={n}")

    for ax, title in zip(axes, ["LGM", "VM20", "VM20bis"]):
        ax.axhline(0.01, color="gray", ls=":", lw=1.2, label="1% level")
        ax.axhline(0.05, color="gray", ls="--", lw=0.8, label="5% level")
        ax.set_xlabel(r"$r / R_e$")
        ax.set_ylabel(r"$|\rho_\mathrm{approx}/\rho_\mathrm{num} - 1|$")
        ax.set_title(f"Relative error: {title} vs numerical")
        ax.legend(fontsize=6, ncol=2)
        ax.set_ylim(1e-4, 3)

    fig.suptitle("Sérsic 3-D deprojection: approximation vs numerical Abel reference",
                 fontsize=12, y=1.01)
    fig.tight_layout()

    # ── Figure 2: 2-D error map for VM20 ──────────────────────────────────
    n_grid = np.logspace(np.log10(0.5), np.log10(10), 40)
    x_grid = np.logspace(-3, 3, 60)
    err_map_lgm = np.full((len(n_grid), len(x_grid)), np.nan)
    err_map_vm20 = np.full_like(err_map_lgm, np.nan)
    err_map_vm20bis = np.full_like(err_map_lgm, np.nan)

    print("Building 2-D error map (this may take a moment)...")
    for j, n in enumerate(n_grid):
        m = SersicModel(re_pc=re_pc, n=n)
        r_arr = np.array(x_grid) * re_pc
        rho_num = m.density_3d_numerical(r_arr)
        if 0.5 <= n <= 10:
            rho_lgm = m.density_3d_LGM(r_arr)
            err_map_lgm[j] = _safe_rel_err(rho_lgm, rho_num)
            rho_vm20 = m.density_3d_VM20(r_arr)
            err_map_vm20[j] = _safe_rel_err(rho_vm20, rho_num)
        if 0.5 <= n <= 3.4:
            x_bis = np.logspace(-4, 3, len(x_grid))
            r_bis = x_bis * re_pc
            rho_num_bis = m.density_3d_numerical(r_bis)
            rho_vm20bis = m.density_3d_VM20bis(r_bis)
            err_map_vm20bis[j] = _safe_rel_err(rho_vm20bis, rho_num_bis)

    fig2, axs = plt.subplots(1, 3, figsize=(17, 4))
    maps = [(err_map_lgm, "LGM", x_grid),
            (err_map_vm20, "VM20", x_grid),
            (err_map_vm20bis, "VM20bis", np.logspace(-4, 3, len(x_grid)))]

    for ax, (err_map, title, xs) in zip(axs, maps):
        im = ax.pcolormesh(np.log10(xs), np.log10(n_grid),
                           np.log10(np.clip(err_map, 1e-5, None)),
                           cmap="RdYlGn_r", vmin=-3, vmax=0)
        ax.set_xlabel(r"$\log_{10}(r/R_e)$")
        ax.set_ylabel(r"$\log_{10}(n)$")
        ax.set_title(f"{title}: $\\log_{{10}}$ relative error")
        fig2.colorbar(im, ax=ax, label=r"$\log_{10}|\rho_\mathrm{approx}/\rho_\mathrm{num}-1|$")

    fig2.suptitle("2-D error map: log10 relative error vs numerical Abel",
                  fontsize=12, y=1.01)
    fig2.tight_layout()

    if output_path:
        stem = pathlib.Path(output_path).stem
        suffix = pathlib.Path(output_path).suffix or ".png"
        p1 = pathlib.Path(output_path).parent / f"{stem}_1d{suffix}"
        p2 = pathlib.Path(output_path).parent / f"{stem}_2d{suffix}"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        print(f"Figures saved to {p1} and {p2}")
    else:
        plt.show()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    main(out)
