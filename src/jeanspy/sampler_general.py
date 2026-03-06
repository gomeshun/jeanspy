import emcee
from emcee import EnsembleSampler
from tqdm import tqdm
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import gzip
from .scatter_matrix import scatter_matrix
from multiprocessing import Pool
from warnings import warn
from numpy import inf


uses_old_emcee = int(emcee.__version__.split(".")[0]) <= 2
if uses_old_emcee:
    raise RuntimeError(print("Your emcee version is {}. Use emcee3.".format(emcee.__version__)))

class Sampler:
    """
    wrapper of emcee.EnsembleSampler. 
    """
    def __init__(self,lnpost,p0,nwalkers=120):
        """
        init
        """
        
        self.lnpost = lnpost
        blobs_dtype = float  # Note: Here dtype must be specified, otherwise an error happens. #[("lnlike",float),]
        self.sampler = EnsembleSampler(nwalkers,p0.shape[1],lnpost,blobs_dtype=blobs_dtype)  # NOTE: dtype must be list of tuple (not tuple of tuple)
        self.p0 = p0
        self.p_last = p0
        self.ndim = p0.shape[1]
        
        
    def reset_sampler(self):
        self.sampler.reset()
        
        
    def sample(self,n_sample,burnin=False,use_pool=False):
        """
        execute mcmc for given iteration steps.
        """
        desc = "burnin" if burnin else "sample"
        
        with Pool() as pool:
            self.sampler.pool = pool if use_pool else None
            iteration = tqdm(self.sampler.sample(self.p_last,iterations=n_sample),total=n_sample,desc=desc)
            for _ret in iteration:
                self.p_last = _ret.coords   # if uses_emcee3 else _ret[0]  # for emcee2
                lnposts     = _ret.log_prob # if uses_emcee3 else _ret[1]  # for emcee2
                iteration.set_postfix(lnpost_min=np.min(lnposts),lnpost_max=np.max(lnposts),lnpost_mean=np.mean(lnposts))
            if burnin:
                self.reset_sampler()
    
    
    def get_chain(self,**kwargs):
        return self.sampler.get_chain(**kwargs)
    
    
    def get_log_prob(self,**kwargs):
        return self.sampler.get_log_prob(**kwargs)
    
    
    def get_blobs(self,**kwargs):
        return self.sampler.get_blobs(**kwargs)
    
    
    def get_last_sample(self,**kwargs):
        return self.sampler.get_last_sample(**kwargs)
    
    
    def _save(self,fname_base):
        np.save(fname_base+"_chain.npy",self.get_chain())
        np.save(fname_base+"_lnprob.npy",self.get_log_prob())
        np.save(fname_base+"_lnlike.npy",self.get_blobs())

    
    def save(self,fname_base):
        '''
        Save MCMC results into "<fname_base>_chain/lnprob/lnlike.npy".
        If fname_base is like "your_directory/your_prefix", create "your_directory" before saving.
        '''
        dirname = os.path.dirname(fname_base)
        if dirname == "":
            self._save(fname_base)
        else:
            if not os.path.isdir(dirname): os.makedirs(dirname, exist_ok=True)
            self._save(fname_base)   
        
        
    def save_pickle(self,fname_base,overwrite=False):
        fname = fname_base+'_.gz'
        if os.path.exists(fname):
            if overwrite : warn(f"{fname} exsits already. It will be overwritten.")
            else         : raise RuntimeError(f"{fname} exsits already. If you want to overwrite it, set \"overwrite=True\".")
        data = pickle.dumps(self)
        with gzip.open(fname, mode='wb') as fp:
            fp.write(data)
            

def load_sampler(fname):
    """
    load Sampler from pickled file.
    """
    with gzip.open(fname, mode='rb') as fp:
        data = fp.read()
    return pickle.loads(data)


class Analyzer:
    
    def __init__(self,*args,**kwargs):
        """
        Initialize an Analyzer instance.
        
        parameters:
            loadtype (default: "pickle"):
                specifing file type from "pickle"/"npy".
                If "pickle" specified, "fname" must be "XXX.gz" saved by Sampler.
                If "npy" specified, "fname_base" must be "XXX"(_chain/_lnprob/_lnlike.npy) saved by Sampler.
                
            args,kwargs: See "Analyzer.load_pickle" or "Analyzer.load_npy_files".
        """
        if "loadtype" not in kwargs:
            kwargs["loadtype"] = "pickle"
            
        loadtype = kwargs["loadtype"]
        del kwargs["loadtype"]
            
        if loadtype == "pickle": 
            self.load_pickle(*args,**kwargs)
        elif loadtype == "npy": 
            self.load_npy_files(*args,**kwargs)
            
        elif loadtype == "sampler":
            self.load_sampler(*args,**kwargs)
        else:
            raise RuntimeError("invalid loadtype")

    
    def load_sampler(self,sampler,keys,n_skipinit=0,n_sep=1,ignore_inf=True,**kwargs):
        self.keys =keys
        self._chain = sampler.get_chain()
        self._lnprobability = sampler.get_log_prob()
        self._lnlike = sampler.get_blobs()
        self.n_skipinit = n_skipinit
        self.n_sep = n_sep
        self.ignore_inf = ignore_inf
    
    def load_pickle(self,fname,keys,n_skipinit=0,n_sep=1,ignore_inf=True,**kwargs):
        sampler = load_sampler(fname)
        self.keys =keys
        self._chain = sampler.get_chain()
        self._lnprobability = sampler.get_log_prob()
        self._lnlike = sampler.get_blobs()
        self.n_skipinit = n_skipinit
        self.n_sep = n_sep
        self.ignore_inf = ignore_inf
    
    
    def load_npy_files(self,fname_base,keys,n_skipinit=0,n_sep=1,ignore_inf=True,**kwargs):
        self.fname_base = fname_base
        self.keys = keys
        self._chain         = np.load(fname_base+"_chain.npy")
        self._lnprobability = np.load(fname_base+"_lnprob.npy")
        self._lnlike = np.load(fname_base+"_lnlike.npy")
        self.n_skipinit = n_skipinit
        self.n_sep = n_sep
        self.ignore_inf = ignore_inf
    
    @property
    def ndim(self):
        return len(self.keys)
    
    @property
    def chain(self):
        return self._chain[self.n_skipinit::self.n_sep,:,:]
    
    @property
    def lnprobability(self):
        return self._lnprobability[self.n_skipinit::self.n_sep,:]
    
    @property
    def lnlike(self):
        return self._lnlike[self.n_skipinit::self.n_sep,:]
    
    @property
    def flatchain(self):
        return self.chain.reshape(-1,self.ndim)
    
    @property
    def flatlnprobability(self):
        return self.lnprobability.reshape(-1)
    
    @property
    def flatlnlike(self):
        return self.lnlike.reshape(-1)
    
    @property
    def df(self):
        _df = DataFrame(self.flatchain,columns=self.keys)
        _df["lnprob"] = self.flatlnprobability
        _df["lnlike"] = self.flatlnlike
        return _df[_df.lnprob>-inf].reset_index(drop=True)
    
    def plot_chain(self,kwargs_subplots={},**kwargs):
        fig,ax = plt.subplots(self.ndim+2,**kwargs_subplots)
        for i in range(self.ndim):
            ax[i].plot(self.chain[:,:,i],**kwargs) # [nwalkers,nsample,ndim]
            ax[i].set_ylabel(self.keys[i])
        ax[self.ndim].plot(self.lnprobability,**kwargs) # [nwalkers,nsample,ndim]
        ax[self.ndim].set_ylabel("lnprob")
        ax[self.ndim+1].plot(self.lnlike,**kwargs) # [nwalkers,nsample,ndim]
        ax[self.ndim+1].set_ylabel("lnlike")
        return fig,ax
        
    
    def plot_lnprob_chain(self,kwargs_subplots={},**kwargs):
        fig,ax = plt.subplots(**kwargs_subplots)
        ax.plot(self.lnprobability,**kwargs) # [nwalkers,nsample,ndim]
        ax.set_ylabel("lnprob")
        return fig,ax
        
    def plot_hist(self,skip=0,n_sep=1,**kwargs):
        self.df.hist(**kwargs)
        
    
    def map_estimater(self):
        _i = self.df.lnprob.idxmax()
        return self.df.iloc[_i]
    

    def scatter_matrix(self,c,plot_axes="lower",hist_kwds=dict(bins=64,histtype="step",color="gray"),**kwargs):
        df = self.df.sort_values(c).reset_index(drop=True)
        #print(df)
        return scatter_matrix(df,plot_axes=plot_axes,
                              c=df[c].values,
                              hist_kwds = hist_kwds,
                              **kwargs)
                                
    
    
Analyser = Analyzer