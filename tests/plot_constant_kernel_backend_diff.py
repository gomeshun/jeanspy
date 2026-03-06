from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jeanspy.model_numpyro as model_numpyro_mod
from jeanspy.model_numpyro import ConstantAnisotropyModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ConstantAnisotropyModel kernel backend differences.")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--u-min", type=float, default=1.0 + 1e-4)
    parser.add_argument("--u-max", type=float, default=1e16)
    parser.add_argument("--n-u", type=int, default=1024)
    parser.add_argument("--r", type=float, default=100.0)
    parser.add_argument("--out", type=str, default="tests/artifacts/constant_kernel_backend_diff.png")
    args = parser.parse_args()

    u_np = np.geomspace(args.u_min, args.u_max, args.n_u).astype(np.float64)
    u = jnp.asarray(u_np, dtype=jnp.float32)
    r_dummy = jnp.asarray(args.r, dtype=jnp.float32)

    with patch.object(model_numpyro_mod, "_HYP2F1_BACKEND", "scipy"):
        k_scipy = np.asarray(ConstantAnisotropyModel().kernel(u, r_dummy, params={"beta_ani": args.beta}))

    with patch.object(model_numpyro_mod, "_HYP2F1_BACKEND", "jax"):
        k_jax = np.asarray(ConstantAnisotropyModel().kernel(u, r_dummy, params={"beta_ani": args.beta}))

    rel = np.abs(k_jax - k_scipy) / (np.abs(k_scipy) + 1e-30)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(u_np, k_scipy, label="scipy", lw=1.2)
    axes[0].plot(u_np, k_jax, label="jax", lw=1.0, ls="--")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("K(u)")
    axes[0].set_title(f"Constant kernel (beta={args.beta})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(u_np, rel, lw=1.2)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("|jax-scipy| / |scipy|")
    axes[1].set_title("Relative difference")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
