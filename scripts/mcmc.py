import os
# Tell JAX to avoid the GPU memory allocation in advance but allocate memory as needed
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpyro
import jax.random
import jax.numpy as jnp
from numpyro.infer import AIES, MCMC, NUTS
import numpyro.distributions as dist
# Check available JAX devices
print("JAX devices:", jax.devices())
import matplotlib.pyplot as plt
import arviz as az
import jeanspy.model_numpyro as model_numpyro
from jeanspy.model_numpyro import ConstantAnisotropyModel, DSphModel, NFWModel, PlummerModel

model_numpyro._HYP2F1_BACKEND = "jax"

key = jax.random.PRNGKey(123)

key, subkey = jax.random.split(key)

true = {
    "re_pc": 200.0,
    "rs_pc": 1200.0,
    "rhos_Msunpc3": 1e-2,
    "r_t_pc": 8000.0,
    "beta_ani": 0.2,
    "vmem_kms": 0.0,
}

# Define the DSph model with the specified submodels
stellar = PlummerModel()
dm = NFWModel()
ani = ConstantAnisotropyModel()
dsph = DSphModel(submodels={"StellarModel": stellar, "DMModel": dm, "AnisotropyModel": ani})

@jax.jit
def sigmalos2(R_pc, params):
    return dsph.sigmalos2(R_pc, params=params, n_u=192, u_max=1500.0)


# Generate mock data
nR = 1000
R_pc = stellar.sample_R(subkey, nR, re_pc=true["re_pc"])
err = 2.0
err2 = (err * jnp.ones_like(R_pc)) ** 2

s2_true = sigmalos2(R_pc, params=true)
key, subkey = jax.random.split(key)
vlos = true["vmem_kms"] + jnp.sqrt(s2_true + err2) * jax.random.normal(subkey, shape=R_pc.shape)

R_obs = jnp.array(R_pc)
v_obs = jnp.array(vlos)
e_obs = jnp.array(err * jnp.ones_like(R_obs))

# Mock data display
plt.figure(figsize=(5, 3))
plt.errorbar(R_obs, v_obs, yerr=e_obs, fmt="o", ms=4, alpha=0.8)
plt.xscale("log")
plt.title("Mock data: Jeans vlos vs R")
plt.xlabel("R [pc]")
plt.ylabel("vlos [km/s]")
plt.grid(True, alpha=0.3)
plt.show()

# Velocity dispersion proifile (Mock & exact) display
plt.figure(figsize=(5, 3))
_R_obs = jnp.logspace(0,3)
_sig = sigmalos2(_R_obs, params=true)
_sig = jnp.sqrt(_sig)
plt.plot(_R_obs,_sig)
plt.xscale("log")
# plt.ylim(bottom=0)
plt.xlabel("R [pc]")
plt.ylabel("vlos [km/s]")
plt.grid(True, alpha=0.3)
plt.show()

def jeans_model(R_pc, vlos_kms, e_vlos_kms):
    # Forward model for Jeans MCMC sampling.
    # This will be called by the MCMC sampler with different parameters to compute 
    # the likelihood of the observed data.

    # log_re = numpyro.sample("log_re", dist.Normal(jnp.log(200.0), 0.8))
    log_re = jnp.log(200)  # deterministic for testing
    log_rs = numpyro.sample("log_rs", dist.Normal(jnp.log(1200.0), 0.8))
    log_rhos = numpyro.sample("log_rhos", dist.Normal(jnp.log(1e-2), 1.0))
    log_r_t = numpyro.sample("log_r_t", dist.Normal(jnp.log(8000.0), 0.4))
    beta_ani = numpyro.sample("beta_ani", dist.Uniform(-0.5, 0.8))
    vmem_kms = numpyro.sample("vmem_kms", dist.Normal(0.0, 30.0))

    # transform to physical parameters
    re_pc = jnp.exp(log_re)
    rs_pc = jnp.exp(log_rs)
    rhos_Msunpc3 = jnp.exp(log_rhos)
    r_t_pc = jnp.exp(log_r_t)

    # (optional): save parameters for diagnostics
    numpyro.deterministic("re_pc", re_pc)
    numpyro.deterministic("rs_pc", rs_pc)
    numpyro.deterministic("rhos_Msunpc3", rhos_Msunpc3)
    numpyro.deterministic("r_t_pc", r_t_pc)
    # numpyro.deterministic("beta_ani", beta_ani)
    # numpyro.deterministic("vmem_kms", vmem_kms)

    params = {
        "re_pc": re_pc,
        "rs_pc": rs_pc,
        "rhos_Msunpc3": rhos_Msunpc3,
        "r_t_pc": r_t_pc,
        "beta_ani": beta_ani,
        "vmem_kms": vmem_kms,
    }

    s2 = sigmalos2(jnp.asarray(R_pc), params=params)
    s2 = jnp.clip(s2, min=1e-12, max=1e12)
    scale = jnp.sqrt(s2 + jnp.asarray(e_vlos_kms) ** 2)
    numpyro.sample("vlos", dist.Normal(vmem_kms, scale), obs=jnp.asarray(vlos_kms))


# MCMC sampling
num_chains = 2
num_warmup = 50
num_samples = 2000
num_epochs = 10

num_chains = 2  # num_chains>=2 is required for az.summary
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num_chains)

sampler = NUTS(jeans_model)
mcmc_jeans = MCMC(
    sampler,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,  
    chain_method="parallel",
    progress_bar=False,
    jit_model_args=True  # NOTE: this does not take effect for the case num_chains > 1 and chain_method == 'parallel'
)

idata_concat = None
for i_epoch in range(num_epochs):
    mcmc_jeans.post_warmup_state = mcmc_jeans.last_state  # warmup from the last state of the previous epoch
    mcmc_jeans.run(keys, R_pc=R_obs, vlos_kms=v_obs, e_vlos_kms=e_obs)
    # save chain
    idata = az.from_numpyro(mcmc_jeans)
    if idata_concat is None:
        idata_concat = idata
    else:
        az.concat(idata_concat, idata, dim="draw", inplace=True)
        az.to_netcdf(idata_concat, f"tests/artifacts/jeans_mcmc.nc")
        az.plot_trace(idata_concat, var_names=["log_re", "log_rs", "log_rhos", "log_r_t", "beta_ani", "vmem_kms"])
        plt.savefig(f"tests/artifacts/trace_epoch_{i_epoch+1}.png", dpi=150)
        plt.close()