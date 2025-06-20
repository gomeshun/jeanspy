from typing import Any
import pandas as pd
import numpy as np
import emcee 
import emcee.backends
import torch
from astropy.io import ascii
from astropy.coordinates import SkyCoord,Distance
import astropy.units as u
from tqdm import tqdm as tqdm
from multiprocessing import Pool, cpu_count
import itertools
import os
from pprint import pprint

############################################
# # How to use blobs in emcee
# 
# import emcee
# import numpy as np

# def log_prior(params):
#     return -0.5 * np.sum(params**2)

# def log_like(params):
#     return -0.5 * np.sum((params / 0.1)**2)

# def log_prob(params):
#     lp = log_prior(params)
#     if not np.isfinite(lp):
#         return -np.inf, -np.inf, -np.inf
#     ll = log_like(params)
#     if not np.isfinite(ll):
#         return lp, -np.inf, -np.inf
#     return lp + ll, lp, np.mean(params)

# coords = np.random.randn(32, 3)
# nwalkers, ndim = coords.shape

# # Here are the important lines
# dtype = [("log_prior", float), ("mean", float)]
# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
#                                 blobs_dtype=dtype)

# sampler.run_mcmc(coords, 100)

# blobs = sampler.get_blobs()
# log_prior_samps = blobs["log_prior"]
# mean_samps = blobs["mean"]
# print(log_prior_samps.shape)
# print(mean_samps.shape)

# flat_blobs = sampler.get_blobs(flat=True)
# flat_log_prior_samps = flat_blobs["log_prior"]
# flat_mean_samps = flat_blobs["mean"]
# print(flat_log_prior_samps.shape)
# print(flat_mean_samps.shape)

############################################
# # How to use backends in emcee
# import emcee
# import numpy as np

# np.random.seed(42)

# # The definition of the log probability function
# # We'll also use the "blobs" feature to track the "log prior" for each step
# def log_prob(theta):
#     log_prior = -0.5 * np.sum((theta - 1.0) ** 2 / 100.0)
#     log_prob = -0.5 * np.sum(theta**2) + log_prior
#     return log_prob, log_prior


# # Initialize the walkers
# coords = np.random.randn(32, 5)
# nwalkers, ndim = coords.shape

# # Set up the backend
# # Don't forget to clear it in case the file already exists
# filename = "tutorial.h5"
# backend = emcee.backends.HDFBackend(filename)
# backend.reset(nwalkers, ndim)

# # Initialize the sampler
# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, backend=backend)

# max_n = 100000

# # We'll track how the average autocorrelation time estimate changes
# index = 0
# autocorr = np.empty(max_n)

# # This will be useful to testing convergence
# old_tau = np.inf

# # Now we'll sample for up to max_n steps
# for sample in sampler.sample(coords, iterations=max_n, progress=True):
#     # Only check convergence every 100 steps
#     if sampler.iteration % 100:
#         continue

#     # Compute the autocorrelation time so far
#     # Using tol=0 means that we'll always get an estimate even
#     # if it isn't trustworthy
#     tau = sampler.get_autocorr_time(tol=0)
#     autocorr[index] = np.mean(tau)
#     index += 1

#     # Check convergence
#     converged = np.all(tau * 100 < sampler.iteration)
#     converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
#     if converged:
#         break
#     old_tau = tau
############################################


class Sampler:
    """ wrapper class for emcee.EnsembleSampler
    """

    def __init__(self,model,p0_generator,nwalkers=None,prefix="",reset=False,pool=None,wbic=False,**kwargs):
        """ initialize the sampler.
        
        model: a model class
        p0_generator: a function to generate p0
        nwalkers: number of walkers. default is 2*ndim
        prefix: prefix of the filename
        reset: reset the backend
        **kwargs: keyword arguments for emcee.EnsembleSampler
        """
        self.model = model
        self.ndim = model.ndim
        self.nwalkers = model.ndim * 2 if nwalkers is None else nwalkers
        self.p0_generator = p0_generator
        self.kwargs = kwargs
        self.pool = pool
        blobs_dtype = [ ("lnl",float), *[ (name, float) for name in self.model.prior_names ]]
        print("blobs_dtype:")
        pprint(blobs_dtype)

        # Show some message to check the definition of the 'convert_params' 
        print("Sampler: Please check the following lines to see your parameters are properly converted:")
        if p0_generator is None:
            print("Sampler: p0_generator is None so we skip the conversion of p0. Please make sure to convert p0 properly by yourself.")
        else:
            # p0 = p0_generator(1)
            # params = self.model.convert_params(p0[0])
            p0 = p0_generator(None)
            params = self.model.convert_params(p0)
            print(f"Sampler: p0: {p0}")
            print(f"---->")
            print(f"Sampler: params: ")
            print(params)
            # NOTE: Instead of above lines, we use comparison dataframe
            # Here we note that params is a pandas.Series and p0 is a numpy.ndarray
            conparison = pd.DataFrame({"p0":p0,"params":params})
            print(conparison)
        
        # define filename as a comination of model name and current time
        # NOTE: replace "+" in the model name
        import time
        filename = prefix + "_".join([self.model.name.replace("+","_"),model.dsph_name]) + ".h5"
        print("Sampler: filename:",filename)
        if wbic:
            self.backend_name = "mcmc_wbic"
            # self.backend = emcee.backends.HDFBackend(filename,name="mcmc_wbic")
        else:
            self.backend_name = "mcmc"
            # self.backend = emcee.backends.HDFBackend(filename)
        self.backend = emcee.backends.HDFBackend(filename,name=self.backend_name)
        # self.backend = emcee.backends.Backend()

        if reset:
            self.backend.reset(self.nwalkers, self.ndim)

        file = self.backend.open()
        print("Sampler: backend file", file)
        print("Sampler: backend file.keys", file.keys())
        try:
            print(f"Sampler: backend file[{self.backend_name}].keys", file[self.backend_name].keys())
        except:
            pass
        file.close()
        
        
        if wbic:
            self.log_prob = self.model.lnposterior_wbic
        else:
            self.log_prob = self.model.lnposterior

        self.sampler = emcee.EnsembleSampler(self.nwalkers,
                                             self.ndim,
                                             log_prob_fn=self.log_prob,
                                             blobs_dtype=blobs_dtype,
                                             backend=self.backend,
                                             pool=pool,
                                             **self.kwargs)


    @property
    def filename(self):
        return self.backend.filename

    def run_mcmc(self,iterations,loops,reset=False,**kwargs):
        """ run the sampler.
        p0 is generated by p0_generator.
        save and monitor the chain and lnprob using backends.
        blobs_dtype is obtained by model.

        iterations: number of iterations for each loop
        loops: number of loops
        """
        # Set up the backend
        # Don't forget to clear it in case the file already exists

        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(loops)

        # This will be useful to testing convergence
        old_tau = np.inf

        # Now we'll sample for up to  steps
        for i_loop in range(loops):
            # start mcmc sampling with self.sampler.run_mcmc and pool
            initial_state = self.p0_generator(self.nwalkers) if self.backend.iteration == 0 else None
            # check if initial_state returns finite log_prob
            # if not, raise an error
            if (initial_state is not None) and (not np.all(np.isfinite([self.model.lnposterior(p) for p in initial_state]))):
                mes = []
                mes.append("Sampler: initial_state has non-finite log_prob")
                for p in initial_state:
                    if not np.all(np.isfinite(self.model.lnposterior(p))):
                        mes.append(f"p:{p}")
                        mes.append(f"lnposterior(p):{self.model.lnposterior(p)}")
                # mes.append(f"initial_state:{initial_state}")
                # mes.append(f"lnposterior:{[self.model.lnposterior(p) for p in initial_state]}")
                raise RuntimeError("\n".join(mes))
            # run mcmc
            self.sampler.run_mcmc(initial_state,iterations,
                                    progress=True,
                                    **kwargs)
            
            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = self.sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < self.sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            else:
                print(f"tau:{tau}\titeration:{self.sampler.iteration}")
                old_tau = tau

    def get_blobs(self,flat=False,thin=1,discard=0):
        """ get blobs from the backend.
        """
        return self.backend.get_blobs(flat=flat,thin=thin,discard=discard)
    
    def get_chain(self,flat=False,thin=1,discard=0):
        """ get chain from the backend.
        """
        return self.backend.get_chain(flat=flat,thin=thin,discard=discard)
    
    def get_log_prob(self,flat=False,thin=1,discard=0):
        """ get log_prob from the backend.
        """
        return self.backend.get_log_prob(flat=flat,thin=thin,discard=discard)
    
    def get_last_sample(self):
        """ get the last sample from the backend.
        """
        return self.backend.get_last_sample()
    
    def get_dataframe(self,thin=1,discard=0,with_lnprob=True):
        """ get the dataframe from the backend.
        """
        chain = self.backend.get_chain(flat=True,thin=thin,discard=discard)
        columns = self.model.submodels["FlatPriorModel"].data.index.tolist()
        df = pd.DataFrame(chain,columns=columns)
        if with_lnprob:
            log_prob = self.backend.get_log_prob(flat=True,thin=thin,discard=discard)
            df["lnprob"] = log_prob
        return df


        

import swyft

class DSphSimulator(swyft.Simulator):
    """ Simulator class for dSph model.
    """
    def __init__(self,model):
        super().__init__()
        self.transform_samples = swyft.to_numpy32  # convert samples to numpy array
        self.model = model  # model class for dSph
        self.R_pc = self.model.data["R_pc"].values  # Position of observed stars in pc
        self.dsph_model = self.model.submodels["DSphModel"]  # model class for dSph
        self.vmem_kms_index = self.model.submodels["FlatPriorModel"].data.index.get_loc("vmem_kms")  # index of "vmem_kms" in model.submodels["PriorModel"].data (pandas.Series)
        self.store_dir = "DSphSimulator_samples"
        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir)
        self.store = {}

    def printlog(self,*txt):
        prefix = self.__class__.__name__ + ":"
        print(prefix,*txt)

    def get_store_path(self,fname):
        return os.path.join(self.store_dir,fname)

    def get_vmem_kms(self,p):
        """ get vmem_kms from model.
        """
        return p[self.vmem_kms_index]


    def get_slos(self,p):
        """ get line-of-sight velocity from model.
        """
        params = self.model.convert_params(p)
        self.model.update("all",params)
        slos = self.dsph_model.sigmalos_dequad(self.R_pc)
        return slos

    
    def get_vlos_kms(self,vmem_kms,slos):
        """ get line-of-sight velocity from model.
        """
        return vmem_kms + slos * np.random.randn(len(self.R_pc))


    def build(self,graph):
        """ build the simulator.
        """
        p = graph.node("p", self.model.sample)
        # find "vmem_kms" in model.submodels["PriorModel"].data (pandas.Series)
        # and get the index
        vmem_kms = graph.node("vmem_kms", self.get_vmem_kms, p)
        slos = graph.node("slos_kms", self.get_slos, p)  # shape: (nstars,)
        v_kms = graph.node("vlos_kms", self.get_vlos_kms, vmem_kms, slos)
    
    @property
    def default_zarr_filename(self):
        """ default zarr filename.
        """
        return f"zarr_{self.model.__class__.__name__}_{self.model.dsph_name}"
    

    def init_zarrstore(self, N, chunk_size, targets=None, zarr_filename=None):
        """ initialize zarr store.
        """
        if targets is None:
            targets = {}
        zarr_filename = self.default_zarr_filename if zarr_filename is None else zarr_filename
        self.store[zarr_filename] = swyft.ZarrStore(self.get_store_path(zarr_filename))
        shapes, dtypes = self.get_shapes_and_dtypes(targets)
        self.store[zarr_filename].init(N, chunk_size, shapes, dtypes)

    
    def simulate_in_zarrstore(self,N, chunk_size, batch_size, targets=None, conditions=None, exclude=None, parallel=True, zarr_filename=None):
        """ simulate in store.
        """
        if conditions is None:
            conditions = {}
        if exclude is None:
            exclude = []
        self.init_zarrstore(N, chunk_size, targets, zarr_filename)

        # if (targets is not None) or (conditions is not None):
        #     _sampler = lambda size: self.sample(size,targets,conditions,exclude)
        # else:
        #     _sampler = self
        worker = Worker(self,targets,conditions,exclude)
        store = self.store[zarr_filename]
        
        if parallel:
            self.printlog("Parallel sampling started.")
            n_pool = min( N // batch_size, cpu_count() ) 
            self.printlog("Parallel sampling with",n_pool,"processes.")
            with Pool(n_pool) as pool:
                # NOTE: self.store.simulate(sampler, max_sims=None, batch_size=10)
                pool.starmap(store.simulate, itertools.repeat([worker,None,batch_size],n_pool))
        else:
            store.simulate(worker, batch_size)


class Worker:
    """ Wrappter class passed to swyft.ZarrStore.simulate
    """

    def __init__(self,sampler,targets,conditions,exclude):
        self.sampler = sampler
        self.targets = targets
        self.conditions = conditions
        self.exclude = exclude

    def __call__(self, size):
        return self.sampler.sample(size,self.targets,self.conditions,self.exclude)


class Network(swyft.SwyftModule):
    def __init__(self,model,enable_1dim=True,enable_ndim=False,enable_embedding=False,dropout=0.1):
        """ Define the network architecture 

        NOTE: When using LogRatioEstimator_Ndim, it prepares internal networks for each marginal.
        If you use embedding, the emmbedding layer is shared among all the marginals.
        For example, if there is a very nice embeding for one embedding, it will be shared for the other marginals.
        But is also means that the other marginals will potentially jam the training of this nice embedding,
        so it could be helpful/harmful depending on the situation.

        This is also the case when using ..._1dim and ..._Ndim simultaniously.
        During the training step, the trainer will combine the two loss functions,
        so the combining could be helpful/harmful depending on the situation as well.
        """

        # One (or both) of enable_1dim and enable_ndim must be True.
        assert enable_1dim or enable_ndim

        super().__init__()
        self.model = model
        self.ndim = model.ndim
        self.enable_1dim = enable_1dim
        self.enable_ndim = enable_ndim
        self.enable_embedding = enable_embedding    
        self.dropout = dropout

        # define embedding
        if enable_embedding:
            n_embedding = model.ndim
            self.embedding = torch.nn.Linear(len(model.data), n_embedding)
        
        num_features = self.n_embedding if enable_embedding else len(model.data)
        self.logratios1 = swyft.LogRatioEstimator_1dim(
            num_features = num_features, 
            num_params = model.ndim, 
            varnames = 'p',
            dropout=dropout)
        
        # define marginals
        if enable_ndim:
            # self.marginals = [ (i,j) for i in range(model.ndim) for j in range(model.ndim) if i < j ]
            self.marginals = [(0,1)]
            self.logratios2 = swyft.LogRatioEstimator_Ndim(
                num_features = num_features, 
                marginals = self.marginals, 
                varnames = 'p',
                dropout=dropout)


    def forward(self, A, B):
        """ Define the forward pass of the network.
        """
        A = self.embedding(A['vlos_kms']) if self.enable_embedding else A['vlos_kms']

        logratios = []
        if self.enable_1dim:
            logratios.append(self.logratios1(A, B['p']))
        if self.enable_ndim:
            logratios.append(self.logratios2(A, B['p']))

        return logratios        