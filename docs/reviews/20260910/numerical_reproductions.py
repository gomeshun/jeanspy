"""Reproduce observed review findings at JeansPy commit 719a316.

This is an audit recorder, not a passing correctness-test suite. It records
the current outputs and independent expectations without modifying JeansPy.
Run with the numpyro_cpu and dev extras (mpmath is needed for Evans references).
"""
import importlib.metadata as md
import json
import warnings

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
from scipy.integrate import quad
from numpyro import distributions as dist
from numpyro.handlers import seed, trace
from numpyro.infer.util import log_density

from jeanspy import model as classical
from jeanspy import model_numpyro as functional
from jeanspy.sampler_numpyro import JeansLikelihoodModel, ParameterSpec

jax.config.update('jax_enable_x64', True)
mp.mp.dps = 70
results = {'versions': {name: md.version(name) for name in ['jeanspy','numpy','scipy','jax','numpyro','mpmath']}}
p = dict(re_pc=200., rs_pc=1000., rhos_Msunpc3=.01, r_t_pc=10000., beta_ani=0., vmem_kms=0.)
dm = functional.NFWModel()
dsph = functional.DSphModel(submodels={'StellarModel': functional.PlummerModel(), 'DMModel': dm,
                                     'AnisotropyModel': functional.ConstantAnisotropyModel()})
results['nfw_integer_numeric_mass'] = np.asarray(dm.enclosed_mass(jnp.array([100,1000]), method='numeric', params=p)).tolist()
results['nfw_float_numeric_mass'] = np.asarray(dm.enclosed_mass(jnp.array([100.,1000.]), method='numeric', params=p)).tolist()
results['nfw_analytic_mass'] = np.asarray(dm.enclosed_mass(jnp.array([100.,1000.]), params=p)).tolist()

bad_p = dict(p, rhos_Msunpc3=float('nan'))
bad_likelihood = JeansLikelihoodModel(dsph, [], parameter_postprocess=lambda _:bad_p,
                                      sigmalos2_kwargs={'dm_mass_method':'numeric'})
results['nan_density_numeric_mass'] = np.asarray(dm.enclosed_mass(jnp.array([100.]),method='numeric',params=bad_p)).tolist()
results['nan_density_numeric_likelihood'] = float(log_density(bad_likelihood,
    (jnp.array([100.]),jnp.array([0.]),jnp.array([2.])),{}, {})[0])

likelihood = JeansLikelihoodModel(dsph, [], parameter_postprocess=lambda _:p)
R = jnp.array([50.,100.,300.]); v = jnp.array([1.,2.,3.]); e = jnp.array([2.,2.,2.])
results['likelihood_shapes'] = {}
for name,velocity,error in [('vector',v,e),('column',v[:,None],e),('negative_error',v,-e)]:
    ld, model_trace = log_density(likelihood,(R,velocity,error),{}, {})
    site=model_trace['vlos']
    results['likelihood_shapes'][name]={'log_density':float(ld),
        'log_prob_shape':list(site['fn'].log_prob(site['value']).shape)}

results['zhao_jfactor_cusps'] = []
for g in [1.,1.49,1.5,1.6,2.5]:
    halo=classical.ZhaoModel(rs_pc=1000.,rhos_Msunpc3=.01,r_t_pc=10000.,a=1.,b=4.,g=g)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        value=halo.jfactor_ullio2016(100000.,.5)
    results['zhao_jfactor_cusps'].append({'g':g,'returned':float(value),
        'mathematically_finite':g<1.5,'warnings':[str(w.message) for w in seen]})

results['evans_near_scale_radius'] = []
halo=classical.NFWModel(rs_pc=1000.,rhos_Msunpc3=.01,r_t_pc=10000.)
distance=100000.
for delta in [-1e-5,-1e-6,-1e-7,0.,1e-7,1e-6,1e-5]:
    y=mp.mpf(str(1.+delta)); d=1-y*y
    if y==1: coefficient=mp.pi-mp.mpf(38)/15
    else:
        X=mp.acosh(1/y)/mp.sqrt(1-y*y) if y<1 else mp.acos(1/y)/mp.sqrt(y*y-1)
        coefficient=(2*y*(7*y-4*y**3+3*mp.pi*d**2)+6*(2*d**3-2*d-y**4)*X)/(6*d**2)
    reference=float(coefficient)*classical.C_J*2*np.pi*.01**2*1000**3/distance**2
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        got=float(halo.jfactor_evans2016(distance,np.rad2deg((1.+delta)*1000./distance)))
    results['evans_near_scale_radius'].append({'y_minus_one':delta,'returned':got,
        'mpmath_70_digit_reference':reference,'relative_error':got/reference-1.,
        'warnings':[str(w.message) for w in seen]})

plummer=classical.PlummerModel(re_pc=200.)
uniform=classical.Uniform2dModel(Rmax_pc=200.)
results['spatial_support']={
    'plummer_truncated_density_at_400_for_cutoff_200':float(plummer.density_2d_truncated(400.,200.)),
    'plummer_truncated_total_probability':quad(lambda r:2*np.pi*r*plummer.density_2d_truncated(r,200.),0,np.inf)[0],
    'uniform_density_at_400_for_radius_200':float(uniform.density_2d(np.array(400.))),
    'uniform_cdf_at_400_for_radius_200':float(uniform.cdf_R(np.array(400.))),
}

results['parameter_spec_default_name']={}
for constructor in [ParameterSpec.exp,ParameterSpec.pow10]:
    def model(): constructor('log_re',dist.Normal(0.,1.)).sample()
    try:
        trace(seed(model,jax.random.PRNGKey(0))).get_trace()
        results['parameter_spec_default_name'][constructor.__name__]='success'
    except Exception as ex:
        results['parameter_spec_default_name'][constructor.__name__]=type(ex).__name__+': '+str(ex)

print(json.dumps(results,indent=2,allow_nan=False))
