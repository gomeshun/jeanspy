"""Shared synthetic data and plotting helpers for the public Quickstart."""
# mock-start
import numpy as np
from jeanspy.model import DSphModel, PlummerModel, ZhaoModel, ConstantAnisotropyModel

# This example fixes photometry and the outer halo shape, and fits five parameters.
true_params = dict(re_pc=200., rs_pc=500., rhos_Msunpc3=.1,
                   alpha=1., beta=3., gamma=1., r_t_pc=np.inf, beta_ani=0., vmem_kms=0.)
truth_model = DSphModel(vmem_kms=true_params["vmem_kms"], submodels={
    "StellarModel": PlummerModel(re_pc=true_params["re_pc"]),
    "DMModel": ZhaoModel(**{k: true_params[k] for k in
                           ("rs_pc", "rhos_Msunpc3", "alpha", "beta", "gamma", "r_t_pc")}),
    "AnisotropyModel": ConstantAnisotropyModel(beta_ani=true_params["beta_ani"]),
})
rng = np.random.default_rng(123)
u = rng.uniform(size=32)
R_pc = true_params["re_pc"] * np.sqrt(u / (1-u))  # Projected Plummer radii.
e_vlos_kms = np.full(R_pc.shape, 2.)
sigma2_true = truth_model.sigmalos2(R_pc, n=256, n_kernel=64)
vlos_kms = rng.normal(true_params["vmem_kms"], np.sqrt(sigma2_true + e_vlos_kms**2))
data = dict(R_pc=R_pc, vlos_kms=vlos_kms, e_vlos_kms=e_vlos_kms)
# mock-end
