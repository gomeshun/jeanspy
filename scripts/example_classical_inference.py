#!/usr/bin/env python3
"""Small reproducible classical inference/storage example, not a calibration run.

Run with the base package installed:
    python scripts/example_classical_inference.py --output-dir /tmp/jeanspy-example
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jeanspy.model import get_default_estimation_model
from jeanspy.sampler import Sampler


def run(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=False)
    np.random.seed(55)
    rng = np.random.default_rng(55)
    # These explicit bounds illustrate the synthetic example only. They are
    # not universal dSph priors or substitutes for a scientific prior choice.
    priors = pd.DataFrame(
        {"lower": [-30., 2., 2.5, -3., 3.5, -.3],
         "upper": [30., 2.6, 3.5, -1., 4.5, .3]},
        index=["vmem_kms", "log10_re_pc", "log10_rs_pc",
               "log10_rhos_Msunpc3", "log10_r_t_pc", "bfunc_beta_ani"],
    )
    prior_path = output_dir / "prior.csv"
    priors.to_csv(prior_path)
    data = pd.DataFrame(dict(R_pc=np.geomspace(10., 800., 32),
                             vlos_kms=np.zeros(32), e_vlos_kms=np.full(32, 2.)))
    model = get_default_estimation_model(data, 2.3, .1, config=prior_path)
    truth = np.array([0., 2.3, 3., -2., 4., 0.])
    model.update(model.convert_params(truth))
    sigma2 = model["DSphModel"].sigmalos2(data.R_pc.to_numpy())
    data["vlos_kms"] = rng.normal(truth[0], np.sqrt(sigma2 + data.e_vlos_kms**2))
    data.to_csv(output_dir / "observations.csv", index=False)
    model.reset_data(data)
    assert np.isfinite(model.lnposterior(truth)).all()
    sampler = Sampler(model, model.sample, nwalkers=16, prefix=str(output_dir) + "/")
    sampler.run_mcmc(8, 1, enable_convergence_check=False)
    initial = sampler.get_chain().copy()
    restored_model = get_default_estimation_model(data, 2.3, .1, config=prior_path)
    resumed = Sampler(restored_model, restored_model.sample, nwalkers=16, prefix=str(output_dir) + "/")
    resumed.run_mcmc(4, 1, enable_convergence_check=False)
    np.testing.assert_array_equal(initial, resumed.get_chain()[:8])
    assert resumed.get_chain().shape == (12, 16, 6)
    assert np.isfinite(resumed.get_log_prob()).all()
    result = {"seed": 55, "truth_sampling_coordinates": truth.tolist(),
              "chain_shape": list(resumed.get_chain().shape),
              "finite_log_probability": True, "restart_prefix_preserved": True,
              "interpretation": "Workflow smoke test; no convergence or coverage claim."}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New directory for synthetic data, priors, chain and result")
    run(parser.parse_args().output_dir)
