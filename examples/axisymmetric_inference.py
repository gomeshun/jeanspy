"""Reproducible axisymmetric inference, restart and posterior J/D example.

The defaults deliberately run short chains on eight synthetic stars to exercise
the workflow. They do not establish posterior convergence or calibration.
Repeat the same command/output directory to append to the saved chain.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jeanspy.axisymmetric import AxisymmetricDSphModel
from jeanspy.axisymmetric_inference import AxisymmetricDSphEstimationModel, AxisymmetricKinematicData


FIXED = dict(re_pc=300., q_projected=.8, Q=.8, alpha=2., beta=3., gamma=.5, r_t_pc=3000.)
TRUTH = dict(**FIXED, rs_pc=500., rhos_Msunpc3=.1, beta_z=-.3,
             inclination=float(np.arccos(.3)), vmem_kms=0.)
PRIOR = pd.DataFrame(dict(lower=[-1.3, 2.5, .07, .1, -20.],
                          upper=[-.7, 2.85, .2, .6, 20.]),
                     index=["log10_rhos_Msunpc3", "log10_rs_pc", "bfunc_beta_z",
                            "cos_inclination", "vmem_kms"])


def mock_data(stars, seed):
    rng = np.random.default_rng(seed)
    # Positions selected inside a finite photometric window. The inference
    # conditions on these observed positions rather than fitting their counts.
    probability = rng.uniform(.03, .8, stars)
    radius = TRUTH["re_pc"]*np.sqrt(probability/(1-probability))
    azimuth = rng.uniform(0., 2*np.pi, stars)
    x, y = radius*np.cos(azimuth), radius*np.sin(azimuth)*TRUTH["q_projected"]
    error = np.full(stars, 2.)
    variance = AxisymmetricDSphModel(96, 96, 96).sigmalos2(x, y, params=TRUTH)
    velocity = rng.normal(TRUTH["vmem_kms"], np.sqrt(variance+error**2))
    return AxisymmetricKinematicData(x, y, velocity, error)


def run_classical(args, estimation):
    from jeanspy.sampler import Sampler
    generator = partial(estimation.sample, rng=np.random.default_rng(args.seed+1))
    sampler = Sampler(estimation, generator, nwalkers=2*estimation.ndim+2,
                       prefix=str(args.output_dir/"chain_"))
    resumed = bool(sampler.backend.iteration > 0)
    if not resumed:
        sampler.sampler.random_state = np.random.RandomState(args.seed+2).get_state()
        sampler.burn_in(args.warmup, generator)
    sampler.run_mcmc(args.draws, 1, enable_convergence_check=False)
    frame = sampler.get_dataframe(discard=args.warmup)
    return frame[estimation.p_names_lnprob], dict(resumed=resumed,
        stored_steps=int(sampler.backend.iteration), walkers=sampler.nwalkers,
        mean_acceptance=float(np.mean(sampler.sampler.acceptance_fraction)),
        warmup_steps_to_discard=args.warmup)


def run_numpyro(args, data):
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value
    from jeanspy.axisymmetric_numpyro import AxisymmetricDSphModel as JaxModel
    from jeanspy.sampler_numpyro import AxisymmetricJeansLikelihoodModel, ParameterSpec, NumPyroSampler

    specifications = []
    for name, row in PRIOR.iterrows():
        distribution = dist.Uniform(row.lower, row.upper)
        if name.startswith("log10_"):
            spec = ParameterSpec.pow10(name, distribution, param_name=name[6:])
        elif name.startswith("bfunc_"):
            spec = ParameterSpec(name, distribution, param_name=name[6:], transform=lambda x: 1-10.**x)
        elif name == "cos_inclination":
            spec = ParameterSpec(name, distribution, param_name="inclination", transform=jnp.arccos)
        else:
            spec = ParameterSpec(name, distribution)
        specifications.append(spec)
    likelihood = AxisymmetricJeansLikelihoodModel(JaxModel(args.nodes, args.nodes, args.nodes),
                                                  specifications, fixed_params=FIXED)
    initial = dict(log10_rhos_Msunpc3=-1., log10_rs_pc=float(np.log10(500.)),
                   bfunc_beta_z=float(np.log10(1.3)), cos_inclination=.3, vmem_kms=0.)
    kernel = NUTS(likelihood, max_tree_depth=4, init_strategy=init_to_value(values=initial))
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.draws,
                num_chains=1, progress_bar=False)
    with NumPyroSampler(mcmc, output_dir=args.output_dir/"numpyro",
                        storage_backend=args.storage, async_writes=False) as sampler:
        result = sampler.run(jax.random.PRNGKey(args.seed), **data.as_kwargs())
        posterior = sampler.load_samples()["posterior"].dataset
        frame = pd.DataFrame({name: posterior[name].values.reshape(-1) for name in PRIOR.index})
        diagnostics = dict(resumed=result.resumed, stored_draws=len(frame),
                            divergences_this_run=int(np.sum(mcmc.get_extra_fields()["diverging"])))
    return frame, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["classical", "numpyro"], default="classical")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stars", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--draws", type=int, default=12)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--storage", choices=["h5netcdf", "zarr", "netcdf4"], default="h5netcdf")
    args = parser.parse_args()
    if args.stars < 2 or args.draws < 1 or args.warmup < 1:
        parser.error("stars >= 2 and positive warmup/draws are required")
    data = mock_data(args.stars, args.seed)
    estimation = AxisymmetricDSphEstimationModel(data, PRIOR, fixed_params=FIXED,
                    dsph_model=AxisymmetricDSphModel(args.nodes, args.nodes, args.nodes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, diagnostics = (run_classical(args, estimation) if args.backend == "classical"
                           else run_numpyro(args, data))
    frame.to_csv(args.output_dir/"posterior.csv", index=False)
    pd.DataFrame(data.as_kwargs()).to_csv(args.output_dir/"observations.csv", index=False)
    PRIOR.to_csv(args.output_dir/"prior.csv")
    # A small deterministic subset illustrates postprocessing a chain from
    # either backend with the same finite-cone factors and physical parameters.
    derived = []
    for index in np.unique(np.linspace(0, len(frame)-1, min(4, len(frame)), dtype=int)):
        physical = estimation.convert_params(frame.iloc[index].to_numpy())
        derived.append(dict(row=int(index),
            J_GeV2_cm_minus5=estimation.dsph_model.jfactor(80000., .5, params=physical),
            D_GeV_cm_minus2=estimation.dsph_model.dfactor(80000., .5, params=physical)))
    pd.DataFrame(derived).to_csv(args.output_dir/"derived_factors.csv", index=False)
    summary = dict(backend=args.backend, seed=args.seed, stars=args.stars, nodes=args.nodes,
                    truth=TRUTH, diagnostics=diagnostics,
                    note="Short workflow demonstration; convergence and scientific calibration are not established.")
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
