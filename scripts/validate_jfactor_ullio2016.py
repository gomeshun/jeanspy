"""Visual validation of Ullio & Valli (2016) J-factor implementations.

Run from the repository root after installing JeansPy in editable mode:

    python scripts/validate_jfactor_ullio2016.py

The script scans ROI for representative NFW and cored Zhao profiles, compares
the full finite-ROI geometry with the legacy/simple spherical-aperture
approximation, and independently checks selected points using a direct
solid-angle + line-of-sight integral.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad

from jeanspy.model import C_J, NFWModel, ZhaoModel


def direct_los_jfactor(model, dist_pc, roi_deg):
    """Independent direct dOmega d(LOS) reference calculation.

    The substitutions b=b_max*u^2 and z=b*tan(t) make this stable even for the
    NFW central cusp.
    """
    r_t_pc = float(model.params.r_t_pc)
    b_max_pc = min(dist_pc * np.sin(np.deg2rad(roi_deg)), r_t_pc)

    def los_integral(b_pc):
        z_max_pc = np.sqrt(max(r_t_pc**2 - b_pc**2, 0.0))
        t_max = np.arctan2(z_max_pc, b_pc)

        def integrand(t):
            cos_t = np.cos(t)
            r_pc = b_pc / cos_t
            rho = float(model.mass_density_3d(r_pc))
            return rho**2 * b_pc / cos_t**2

        value, _ = quad(
            integrand, 0.0, t_max, epsabs=0.0, epsrel=2e-9, limit=400
        )
        return 2.0 * value

    def integrand_u(u):
        if u == 0.0:
            return 0.0
        b_pc = b_max_pc * u**2
        db_du = 2.0 * b_max_pc * u
        domega_db = (
            2.0 * np.pi * b_pc
            / (dist_pc * np.sqrt(dist_pc**2 - b_pc**2))
        )
        return domega_db * los_integral(b_pc) * db_du

    value, _ = quad(
        integrand_u, 0.0, 1.0, epsabs=0.0, epsrel=2e-7, limit=400
    )
    return C_J * value


def main():
    output_dir = Path("validation")
    output_dir.mkdir(exist_ok=True)

    dist_pc = 80_000.0
    r_s_pc = 500.0
    rho_s = 0.1
    r_t_pc = 1_000.0
    theta_t = np.rad2deg(np.arcsin(r_t_pc / dist_pc))

    profiles = {
        "NFW": NFWModel(
            rs_pc=r_s_pc, rhos_Msunpc3=rho_s, r_t_pc=r_t_pc
        ),
        "Cored Zhao": ZhaoModel(
            rs_pc=r_s_pc,
            rhos_Msunpc3=rho_s,
            a=1.0,
            b=3.0,
            g=0.0,
            r_t_pc=r_t_pc,
        ),
    }

    rois = np.geomspace(0.01, 1.0, 80)
    check_rois = [0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 1.0]

    rows = []
    check_rows = []

    for name, model in profiles.items():
        for roi_deg in rois:
            full = model.jfactor_ullio2016(dist_pc, roi_deg)
            simple = model.jfactor_ullio2016_simple(dist_pc, roi_deg)
            rows.append(
                {
                    "profile": name,
                    "roi_deg": roi_deg,
                    "j_full": full,
                    "j_simple": simple,
                    "full_over_simple": full / simple,
                }
            )

        for roi_deg in check_rois:
            full = model.jfactor_ullio2016(dist_pc, roi_deg)
            direct = direct_los_jfactor(model, dist_pc, roi_deg)
            check_rows.append(
                {
                    "profile": name,
                    "roi_deg": roi_deg,
                    "j_full": full,
                    "j_direct_los": direct,
                    "relative_difference": full / direct - 1.0,
                }
            )

    df = pd.DataFrame(rows)
    checks = pd.DataFrame(check_rows)
    df.to_csv(output_dir / "jfactor_ullio2016_roi_scan.csv", index=False)
    checks.to_csv(output_dir / "jfactor_ullio2016_direct_check.csv", index=False)

    plt.rcParams["svg.fonttype"] = "none"

    fig, ax = plt.subplots()
    for name in profiles:
        d = df[df["profile"] == name]
        ax.loglog(d["roi_deg"], d["j_full"], label=f"{name}: full")
        ax.loglog(
            d["roi_deg"], d["j_simple"], linestyle="--", label=f"{name}: simple"
        )
    ax.axvline(theta_t, linestyle=":", label=r"$R_{\max}=r_t$")
    ax.set_xlabel("ROI [deg]")
    ax.set_ylabel(r"$J\ [\mathrm{GeV}^2\,\mathrm{cm}^{-5}]$")
    ax.set_title("Ullio full geometry vs spherical-aperture approximation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "jfactor_ullio2016_roi_scan.svg")
    plt.close(fig)

    fig, ax = plt.subplots()
    for name in profiles:
        d = df[df["profile"] == name]
        ax.semilogx(d["roi_deg"], d["full_over_simple"], label=name)
    ax.axhline(1.0, linestyle=":")
    ax.axvline(theta_t, linestyle=":", label=r"$R_{\max}=r_t$")
    ax.set_xlabel("ROI [deg]")
    ax.set_ylabel("full / simple")
    ax.set_title("Projected outer-shell contribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "jfactor_ullio2016_full_simple_ratio.svg")
    plt.close(fig)

    print(checks.to_string(index=False))
    print(f"\nProjected truncation angle: {theta_t:.6f} deg")


if __name__ == "__main__":
    main()
