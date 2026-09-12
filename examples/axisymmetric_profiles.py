"""Run: python examples/axisymmetric_profiles.py [--output profiles.png].

Illustrative parameters, not a fit to an observed dwarf. NumPy-only models;
matplotlib is needed only when --output is supplied.
"""
import argparse
from dataclasses import replace

import numpy as np
from jeanspy.axisymmetric import AxisymmetricJeans, PlummerTracer, ZhaoHalo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional plot filename")
    args = parser.parse_args()
    model = AxisymmetricJeans(
        PlummerTracer(a_pc=300, q=.65),
        ZhaoHalo(rho_s=.1, r_s=500, Q=.7, alpha=2, beta=3, gamma=1),
        beta_z=-.2, inclination=np.deg2rad(70),
    )
    radius = np.geomspace(10, 1500, 16)
    major = np.sqrt(model.los_second_moment(radius, 0))
    minor = np.sqrt(model.los_second_moment(0, radius))
    fine = replace(model, n_force=128, n_vertical=128, n_los=128)
    checks = fine.los_second_moment(radius[[0, 7, -1]], 0)
    relative = np.max(np.abs(major[[0, 7, -1]]**2/checks - 1))
    print(f"Maximum 96/128-node relative second-moment difference: {relative:.3g}")
    print("radius_pc,major_sigma_kms,minor_sigma_kms")
    for row in zip(radius, major, minor):
        print(",".join(f"{v:.8g}" for v in row))
    if args.output:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        ax.plot(radius, major, label="Major axis")
        ax.plot(radius, minor, label="Minor axis")
        ax.set(xlabel="Projected radius [pc]", ylabel="LOS dispersion [km/s]",
               title="Axisymmetric Jeans model (illustrative)", xscale="log")
        ax.legend()
        fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
