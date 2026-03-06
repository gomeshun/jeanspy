from memfg.modki17 import modKI17_memonly, modKI17_photometry
from memfg.dsph_model import NFW_model, plummer_model, dSph_model
from memfg.dsph_model_general import Baes_Anisotropy_Model
from memfg.dsph_model_general import dSph_model as dSph_model_general
from pandas import DataFrame as DF
from pandas import read_csv 
from astropy.units import km, second,pc, deg
import numpy as np
from numpy import nan,log2
from emcee import EnsembleSampler
from emcee import __version__ as emcee__version__ 
from scipy.interpolate import interp1d
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm

emcee_major_version = emcee__version__.split(".")[0]

class LnLike_photo:
    """
    likelihood class to estimate half-right-radius (re_pc).
    """
    
    _keys = ["log10_re_pc"]
    
    def __init__(self,Rs_pc):
        """
        initialize model and store observables.
        """
        self.Rs = Rs_pc
        self.mdl = plummer_model(re_pc=nan)
        self.reset_keys(LnLike_photo._keys)
        
    def keys(self):
        """
        define keys of parameter dict passed to self.mdl
        """
        return self._key
    
    def reset_keys(self,keys,verbose=True):
        self._keys = keys
        if verbose:
            print(self._keys)
        
    def model_update(self,p):
        """
        pass parameter p (array) for self.mdl to calculate model functions.
        
        p = [log10_re_pc]
        """
        keys = np.array(["re_pc"])
        
        sr = pd.Series(p,index=keys)
        sr["re_pc"] = np.power(10,sr["re_pc"])
        
        self.mdl.update(dict(sr))
        
    def __call__(self,p):
        """
        return log-likelihood + log-prior (flat).
        """
        self.model_update(p)
        likelis = 2*np.pi*self.Rs*self.mdl.density_2d(self.Rs)
        return np.sum(np.log(likelis))
    
    
    

class LnLike:
    """
    main class to estimate DM parameters (\rho_{DM}(r)) and anisotropy.
    """
    _keys = ["re_pc","rs_pc","rhos_Msunpc3","a","b","g","anib"]
    
    def __init__(self,Rs,vs):
        """
        initialize self.model and store observables.
        """
        self.Rs = Rs  # pc
        self.vs = vs  # km/s
        mem = plummer_model(re_pc=nan)
        dm = NFW_model(a=nan,b=nan,g=nan,rs_pc=nan,rhos_Msunpc3=nan,R_trunc_pc=np.inf)
        self.mdl = dSph_model(submodels_dict={"stellar_model":mem,"DM_model":dm},anib=nan)
        self.reset_keys(LnLike._keys)
        
    def keys(self):
        """
        define keys of parameter dict passed to self.mdl
        """
        return self._keys
    
    def reset_keys(self,keys,verbose=True):
        self._keys = keys
        if verbose:
            print(self._keys)
        
    def set_lnprior_func(self,lnprior_func):
        """
        introduce lnprior function (gaussian prior, etc.)
        """
        self._lnprior_func = lnprior_func
        
    def model_update(self,p):
        """
        pass parameter p (array) for self.mdl to calculate model functions.
        array p must be ordered as self.keys
        
        where "mlog10_1manib" denotes -log10(1-anib).
        """
        keys = self.keys
        
        p_dict = dict(zip(keys(),p))
        p_dict["rhos_Msunpc3"] = np.power(10,p_dict["rhos_Msunpc3"])
        p_dict["re_pc"] = np.power(10,p_dict["re_pc"])
        p_dict["rs_pc"] = np.power(10,p_dict["rs_pc"])
        p_dict["anib"] = 1-np.power(10,-p_dict["anib"])
        
        self.mdl.update(p_dict)
    
    
    def __call__(self,p,reflesh=True):
        """
        return log-likelihood + log-prior (flat prior + given function by self.set_lnprior_func)
        
        p = ["log10_re_pc","log10_rs_pc","log10_rhos_Msunpc3","a","b","g","mlog101manib"]
        """
        
        lnp = self.lnprior(p)
        if np.isinf(lnp):
            return -np.inf
        
        sigmalos = self.sigmalos(p)
        lnl = np.sum(norm.logpdf(self.vs,0,sigmalos))
        
        if reflesh:
            self.reflesh()

        return lnl + lnp
    
    
    def sigmalos(self,p,n_interp=64):
        """
        return "sigma_los" for "self.Rs".
        scipy.interpolation.interp1d is adopted to accelerate computing.
        
        self.mdl.sigmalos_dequad: calculate sigma_los by using double-exponential (tanh-sinh) quadrature. 
        """
        self.model_update(p)
        _self_log10_Rs = np.log10(self.Rs)
        
        _Rs = np.logspace(np.min(_self_log10_Rs),np.max(_self_log10_Rs),n_interp)
        _log10_Rs = np.log10(_Rs)
        
        _sigmalos = self.mdl.sigmalos_dequad(_Rs,ignore_nan=True)
        _sigmalos_log10_R = interp1d(_log10_Rs,_sigmalos)
        
        return _sigmalos_log10_R(_self_log10_Rs)
        
        
    def reflesh(self):
        """
        reflesh model parameters to nan.
        """
        p0 =dict(re_pc=nan,a=nan,b=nan,g=nan,rs_pc=nan,rhos_Msunpc3=nan,anib = nan)
        self.mdl.update(p0)
        
        
    def lnprior(self,p):
        """
        return log-prior value.
        flat-prior (hard-cut of parameter space) range is defined by hard-coding.
        """
        log10_re_pc,log10_rs_pc,log10_rhos_Msunpc3,a,b,g,mlog101manib = p
        c = []
        if (0 <= log10_re_pc <= 4) & (-4 <= log10_rs_pc <= 4) & (0.5 <= a <= 3.0) & (3 < b <= 10) & (0.0 <= g <= 1.2) & (-1 <= mlog101manib <= 1):  # if in hard-cut range,
            return self._lnprior_func(p)
        else:
            return -np.inf
        

class LnLike_general:
    """
    main class to estimate DM parameters (\rho_{DM}(r)) and anisotropy.
    """
    _keys = ["re_pc","rs_pc","rhos_Msunpc3","a","b","g","beta_0","beta_inf","eta","r_a"]
    
    def __init__(self,Rs,vs):
        """
        initialize self.model and store observables.
        """
        self.Rs = Rs  # pc
        self.vs = vs  # km/s
        mem = plummer_model(re_pc=nan)
        dm = NFW_model(a=nan,b=nan,g=nan,rs_pc=nan,rhos_Msunpc3=nan,R_trunc_pc=np.inf)
        ani = Baes_Anisotropy_Model(beta_0 = nan, beta_inf=nan,eta=nan,r_a=nan)
        self.mdl = dSph_model_general(submodels_dict={"stellar_model":mem,"DM_model":dm,"Anisotropy_Model":ani})
        self.reset_keys(LnLike_general._keys)
        
    def keys(self):
        """
        define keys of parameter dict passed to self.mdl
        """
        return self._keys
    
    def reset_keys(self,keys,verbose=True):
        self._keys = keys
        if verbose:
            print("keys -> {}".format(self._keys))
        
    def set_lnprior_func(self,lnprior_func):
        """
        introduce lnprior function (gaussian prior, etc.)
        """
        self._lnprior_func = lnprior_func
        
        
    def model_update(self,p):
        """
        pass parameter p (array) for self.mdl to calculate model functions.
        array p must be ordered as self.keys
        
        where "mlog10_1manib" denotes -log10(1-anib).
        """
        keys = self.keys
        
        p_dict = dict(zip(keys(),p))
        
        p_dict["rhos_Msunpc3"] = 10**(p_dict["rhos_Msunpc3"])
        p_dict["re_pc"]        = 10**(p_dict["re_pc"])
        p_dict["rs_pc"]        = 10**(p_dict["rs_pc"])
        p_dict["beta_0"]       = log2(p_dict["beta_0"])
        p_dict["beta_inf"]     = log2(p_dict["beta_inf"])
        p_dict["r_a"]          = 10**(p_dict["r_a"])
        
        self.mdl.update(p_dict)
    
    
    def __call__(self,p,reflesh=True):
        """
        return log-likelihood + log-prior (flat prior + given function by self.set_lnprior_func)
        
        p = ["log10_re_pc","log10_rs_pc","log10_rhos_Msunpc3","a","b","g",...]
        """
        
        lnp = self.lnprior(p)
        if np.isinf(lnp):
            return -np.inf
        
        sigmalos = self.sigmalos(p)
        lnl = np.sum(norm.logpdf(self.vs,0,sigmalos))
        
        if reflesh:
            self.reflesh()

        return lnl + lnp
    
    
    def sigmalos(self,p,n_interp=64):
        """
        return "sigma_los" for "self.Rs".
        scipy.interpolation.interp1d is adopted to accelerate computing.
        
        self.mdl.sigmalos_dequad: calculate sigma_los by using double-exponential (tanh-sinh) quadrature. 
        """
        self.model_update(p)
        _self_log10_Rs = np.log10(self.Rs)
        
        _Rs = np.logspace(np.min(_self_log10_Rs),np.max(_self_log10_Rs),n_interp)
        _log10_Rs = np.log10(_Rs)
        
        _sigmalos = self.mdl.sigmalos_dequad(_Rs,n=128,ignore_RuntimeWarning=True)  # maximum accuracy is achived when n=512. When n=128, relative error is less than 1 %.
        _sigmalos_log10_R = interp1d(_log10_Rs,_sigmalos)
        
        return _sigmalos_log10_R(_self_log10_Rs)
        
        
    def reflesh(self):
        """
        reflesh model parameters to nan.
        """
        # p = ["re_pc","rs_pc","rhos_Msunpc3","a","b","g","beta_0","beta_inf","eta","r_a"]
        p0 =dict(re_pc=nan,a=nan,b=nan,g=nan,rs_pc=nan,rhos_Msunpc3=nan,
                 beta_0 = nan,beta_inf=nan,eta=nan,r_a=nan)
        self.mdl.update(p0)
        
        
    def lnprior(self,p):
        """
        return log-prior value.
        flat-prior (hard-cut of parameter space) range is defined by hard-coding.
        """
        # p = ["re_pc","rs_pc","rhos_Msunpc3","a","b","g","beta_0","beta_inf","eta","r_a"]
        log10_re_pc,log10_rs_pc,log10_rhos_Msunpc3,a,b,g,pow2_beta_0,pow2_beta_inf,eta,log10_r_a = p
        c = []
        if (0 <= log10_re_pc <= 4) & (-4 <= log10_rs_pc <= 4) & (0.5 <= a <= 3.0) & (3 < b <= 10) & (0.0 <= g <= 1.2) & (0 <= pow2_beta_0 <= 1) & (0 <= pow2_beta_inf <= 2) & (1 < eta < 10) & (-2.55 + 3 <= log10_r_a <= 1.5 + 3):  # if in hard-cut range,
            return self._lnprior_func(p)
        else:
            return -np.inf
        

        
class Sampler:
    """
    wrapper of emcee.EnsembleSampler. 
    """
    def __init__(self,lnpost,p0,keys,nwalkers=120):
        self.lnpost = lnpost
        self.sampler = EnsembleSampler(nwalkers,p0.shape[1],lnpost,threads=15)
        self.p0 = p0
        self.p = p0
        self.keys = keys
        self.ndim = len(keys)
        
    def reset_sampler(self):
        self.sampler.reset()
        
    def sample(self,n_sample,burnin=False):
        """
        execute mcmc for given iteration steps.
        """
        desc = "burnin" if burnin else "sample"
        iteration = tqdm(self.sampler.sample(self.p,iterations=n_sample),total=n_sample,desc=desc)
        for _ret in iteration:
            self.p = _ret[0] if emcee_major_version == "2" else _ret.coords
            lnposts = _ret[1]
            iteration.set_postfix(
                lnpost_min=f"{np.min(lnposts):.5e}",  #np.min(lnposts),
                lnpost_max=f"{np.max(lnposts):.5e}",  #np.max(lnposts),
                lnpost_mean=f"{np.mean(lnposts):.5e}"  #np.mean(lnposts)
            )
        if burnin:
            self.reset_sampler()
    
    
    @property
    def df(self):
        """
        convert sampler.chain into pandas.DataFrame for convenience.
        """
        _df = DF(self.sampler.flatchain)
        _df = _df.rename(columns={ i: key for i,key in enumerate(self.keys)} )
        _df["lnpost"] = self.sampler.flatlnprobability
        return _df
        
    
    def save_chain(self,fname):
        self.df.to_pickle(fname)
        
    
    def plot_chain(self,kwargs_subplots={},**kwargs):
        fig,ax = plt.subplots(self.ndim+1,**kwargs_subplots)
        for i in range(self.ndim):
            ax[i].plot(self.sampler.chain[:,:,i].T,**kwargs) # [nwalkers,nsample,ndim]
            ax[i].set_ylabel(self.keys[i])
        ax[self.ndim].plot(self.sampler.lnprobability.T,**kwargs) # [nwalkers,nsample,ndim]
        ax[self.ndim].set_ylabel("lnpost")
        
        
    def plot_hist(self,**kwargs):
        self.df.hist(**kwargs)
        
        
    def map_estimater(self):
        _i = self.df.lnpost.idxmax()
        return self.df.iloc[i]