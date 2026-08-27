#!/usr/bin/env python3
"""Benchmark spherical Sérsic deprojection methods against numerical Abel inversion.

The numerical ``SersicModel.density_3d_numerical`` result is the reference.
This script compares:

* LGM (legacy analytical approximation),
* VM20 (published Vitral & Mamon 2020 coefficients),
* the current JeansPy VM20bis-style *independent refit*,
* Simonneau & Prada (2004; SP04),
* a VM21-style hybrid: VM20bis-style refit for n <= 3.4 and SP04 above it,
* Ciotti, De Deo & Pellegrini (2025), independently reconstructed from the
  asymptotic formula plus the luminosity-conservation condition for p.

Important provenance note
-------------------------
``density_3d_VM20bis`` in the current PR uses coefficients independently
re-fitted inside JeansPy because the historical coefficient repository cited by
Vitral & Mamon (2021) could not be retrieved.  The plots therefore label this
method ``VM20bis-style refit`` rather than presenting its coefficients as the
authors' official VM20bis table.

Usage
-----
    python scripts/compare_sersic_deprojection.py
    python scripts/compare_sersic_deprojection.py output.png

When an output path is supplied, ``*_1d.png`` and ``*_2d.png`` are written.
The script also prints representative maximum/RMS errors and rough evaluation
costs relative to the numerical Abel reference.
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jeanspy._sersic_deprojection import (  # noqa: E402
    ciotti2025_density,
    ciotti2025_matching_p,
    sp04_density,
)
from jeanspy.model import SersicModel  # noqa: E402


RE_PC = 100.0
REPRESENTATIVE_N = [0.5, 0.56, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 10.0]


def _safe_rel_err(rho_approx, rho_ref):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rel = np.abs(np.asarray(rho_approx) / np.asarray(rho_ref) - 1.0)
    return np.where(np.isfinite(rel), rel, np.nan)


def _ciotti_supported(n):
    return n == 0.5 or n == 1.0 or n >= 0.55


def _evaluate(name, model, x):
    r = np.asarray(x) * RE_PC
    n = float(model.params.n)

    if name == "LGM":
        if not (0.5 <= n <= 10):
            return None
        return model.density_3d_LGM(r)

    if name == "VM20":
        if not (0.5 <= n <= 10) or np.min(x) < 1e-3 or np.max(x) > 1e3:
            return None
        return model.density_3d_VM20(r)

    if name == "VM20bis-style refit":
        if not (0.5 <= n <= 3.4) or np.min(x) < 1e-4 or np.max(x) > 1e3:
            return None
        return model.density_3d_VM20bis(r)

    if name == "SP04":
        if n <= 1.0:
            return None
        return sp04_density(r, re_pc=RE_PC, n=n, b=model.b)

    if name == "VM21-style hybrid":
        if not (0.5 <= n <= 10) or np.min(x) < 1e-4 or np.max(x) > 1e3:
            return None
        if n <= 3.4:
            return model.density_3d_VM20bis(r)
        return sp04_density(r, re_pc=RE_PC, n=n, b=model.b)

    if name == "Ciotti+2025":
        if not _ciotti_supported(n):
            return None
        return ciotti2025_density(r, re_pc=RE_PC, n=n, b=model.b)

    raise ValueError(name)


def _method_domain_x(name):
    if name in {"VM20bis-style refit", "VM21-style hybrid", "Ciotti+2025", "SP04", "LGM"}:
        return np.logspace(-4, 3, 150)
    return np.logspace(-3, 3, 150)


def _plot_1d(methods):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
    axes = axes.ravel()
    cmap = plt.get_cmap("tab10")

    for ax, name in zip(axes, methods):
        x = _method_domain_x(name)
        for i, n in enumerate(REPRESENTATIVE_N):
            model = SersicModel(re_pc=RE_PC, n=n, deprojection_method="numerical")
            rho_approx = _evaluate(name, model, x)
            if rho_approx is None:
                continue
            rho_num = model.density_3d_numerical(x * RE_PC)
            err = _safe_rel_err(rho_approx, rho_num)
            ax.loglog(x, err, lw=1.15, color=cmap(i % 10), label=f"n={n:g}")

        ax.axhline(1e-3, color="gray", ls="-.", lw=0.8, label="0.1%")
        ax.axhline(1e-2, color="gray", ls=":", lw=1.0, label="1%")
        ax.axhline(5e-2, color="gray", ls="--", lw=0.8, label="5%")
        ax.set_xlabel(r"$r/R_e$")
        ax.set_ylabel(r"$|\rho_\mathrm{approx}/\rho_\mathrm{num}-1|$")
        ax.set_title(name)
        ax.set_ylim(1e-5, 3.0)
        ax.legend(fontsize=6, ncol=2)

    for ax in axes[len(methods):]:
        ax.axis("off")

    fig.suptitle("Sérsic 3-D deprojection error relative to numerical Abel inversion")
    fig.tight_layout()
    return fig


def _plot_2d(methods):
    # Keep this diagnostic reasonably tractable: Abel inversion dominates cost.
    n_grid = np.logspace(np.log10(0.5), np.log10(10.0), 28)
    x_grid = np.logspace(-4, 3, 46)
    maps = {name: np.full((len(n_grid), len(x_grid)), np.nan) for name in methods}

    print("Building 2-D error maps...")
    for j, n in enumerate(n_grid):
        model = SersicModel(re_pc=RE_PC, n=n, deprojection_method="numerical")
        rho_num = model.density_3d_numerical(x_grid * RE_PC)
        for name in methods:
            # VM20 has the narrower inner calibration boundary.
            valid_x = x_grid >= 1e-3 if name == "VM20" else np.ones_like(x_grid, dtype=bool)
            if not np.any(valid_x):
                continue
            try:
                rho = _evaluate(name, model, x_grid[valid_x])
            except ValueError:
                rho = None
            if rho is None:
                continue
            maps[name][j, valid_x] = _safe_rel_err(rho, rho_num[valid_x])

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, name in zip(axes, methods):
        log_err = np.log10(np.clip(maps[name], 1e-5, 1.0))
        im = ax.pcolormesh(
            np.log10(x_grid),
            np.log10(n_grid),
            log_err,
            cmap="RdYlGn_r",
            vmin=-3,
            vmax=0,
            shading="auto",
        )
        ax.set_xlabel(r"$\log_{10}(r/R_e)$")
        ax.set_ylabel(r"$\log_{10}(n)$")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, label=r"$\log_{10}$ relative error")

    for ax in axes[len(methods):]:
        ax.axis("off")

    fig.suptitle("Sérsic deprojection: 2-D relative-error maps")
    fig.tight_layout()
    return fig, maps, n_grid, x_grid


def _print_error_summary(methods):
    print("\nRepresentative accuracy summary")
    print("method                     max rel. error      RMS rel. error")
    print("-------------------------  ------------------  ------------------")

    accum = {name: [] for name in methods}
    for n in REPRESENTATIVE_N:
        model = SersicModel(re_pc=RE_PC, n=n, deprojection_method="numerical")
        x = np.logspace(-3, 2, 55)
        rho_num = model.density_3d_numerical(x * RE_PC)
        for name in methods:
            try:
                rho = _evaluate(name, model, x)
            except ValueError:
                rho = None
            if rho is None:
                continue
            err = _safe_rel_err(rho, rho_num)
            finite = err[np.isfinite(err)]
            if finite.size:
                accum[name].append(finite)

    for name in methods:
        if not accum[name]:
            continue
        values = np.concatenate(accum[name])
        print(f"{name:25s}  {np.max(values):18.6g}  {np.sqrt(np.mean(values**2)):18.6g}")


def _print_runtime_summary(methods):
    # n=2 is inside every method's mathematical/calibration domain, including
    # the current VM20bis-style refit, which makes the rough timing comparable.
    n = 2.0
    model = SersicModel(re_pc=RE_PC, n=n, deprojection_method="numerical")
    x = np.logspace(-3, 2, 100)
    r = x * RE_PC

    # Warm the Ciotti p root cache so the approximation-evaluation time is not
    # conflated with the one-off luminosity-conservation solve.
    ciotti2025_matching_p(n, model.b)

    t0 = time.perf_counter()
    model.density_3d_numerical(r)
    t_num = time.perf_counter() - t0

    print("\nApproximate evaluation cost at n=2 (100 radii)")
    print(f"numerical Abel             {t_num:.6g} s   (1.0x)")
    for name in methods:
        t0 = time.perf_counter()
        try:
            rho = _evaluate(name, model, x)
        except ValueError:
            rho = None
        elapsed = time.perf_counter() - t0
        if rho is not None:
            print(f"{name:25s} {elapsed:.6g} s   ({elapsed / t_num:.3g}x)")


def main(output_path=None):
    methods = [
        "LGM",
        "VM20",
        "VM20bis-style refit",
        "SP04",
        "VM21-style hybrid",
        "Ciotti+2025",
    ]

    fig1 = _plot_1d(methods)
    fig2, _, _, _ = _plot_2d(methods)
    _print_error_summary(methods)
    _print_runtime_summary(methods)

    if output_path:
        path = pathlib.Path(output_path)
        suffix = path.suffix or ".png"
        p1 = path.parent / f"{path.stem}_1d{suffix}"
        p2 = path.parent / f"{path.stem}_2d{suffix}"
        fig1.savefig(p1, dpi=160, bbox_inches="tight")
        fig2.savefig(p2, dpi=160, bbox_inches="tight")
        print(f"\nFigures saved to {p1} and {p2}")
    else:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
