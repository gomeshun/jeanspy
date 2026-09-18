from typing import Union, Callable
import json
import h5py
import pandas as pd
import numpy as np
import emcee
import emcee.backends
import logging
from multiprocessing import Pool, cpu_count
import itertools
import os
from ._sampling_identity import IDENTITY_FORMAT, fingerprint, software_identity, source_provenance


__all__ = ["Sampler"]

# Module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

class Sampler:
    r"""wrapper class for emcee.EnsembleSampler

    Notes
    -----
    **Inputs and units.** model supplies posterior/blobs and parameter names;
    ``p0_generator`` provides starting positions; nwalkers is an ensemble size;
    prefix sets output filename; reset=True deliberately resets the requested
    store; pool controls parallel evaluation. ``run_mcmc`` receives the run
    length/options in its real signature.

    **Returns and shape.** HDF5 chain and diagnostic blobs; ``get_chain``
    returns (draw,walker,parameter), or flattened draws.
    ``get_log_prob``/``get_blobs`` use corresponding draw/walker axes.
    ``get_dataframe`` returns a pandas table.

    **Validity.** Workers share a stateful model only through the supported
    process/storage setup. Use independent ensembles for R-hat; interacting
    walkers are not independent chains.

    **Errors.** Incompatible persisted analysis identity raises before
    continuing. Missing/malformed state, invalid parameter conversion or storage
    failures raise.

    **Backend.** Python/emcee host sampler around NumPy/SciPy CPU likelihoods.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_inference.py``; complete tutorials for
    production diagnostics.
    """
    def __init__(self, model, p0_generator, nwalkers=None, prefix="", reset=False, pool=None, wbic=False, **kwargs):
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
        self.wbic = bool(wbic)
        self.log_prob = self.model.lnposterior_wbic if wbic else self.model.lnposterior
        self.logger = logger.getChild(self.__class__.__name__)
        blobs_dtype = [("lnl", float), *[(name, float) for name in self.model.prior_names]]
        self.logger.info("blobs_dtype: %s", blobs_dtype)

        # Keep a stable filename so another sampler can resume the stored chain.
        model_name = self.model.name.replace("+", "_")
        dsph_name = getattr(model, "dsph_name", None)
        filename = prefix + model_name + (f"_{dsph_name}" if dsph_name else "") + ".h5"
        self.logger.info("filename: %s", filename)
        self.backend_name = "mcmc_wbic" if wbic else "mcmc"
        self.backend = emcee.backends.HDFBackend(filename, name=self.backend_name)
        # reset file if "reset" is True or if the file does not exist
        identity = self._current_analysis_identity()
        if reset or not self.backend.initialized:
            self.backend.reset(self.nwalkers, self.ndim)
            self._write_analysis_identity(identity)
        else:
            self._check_analysis_identity(identity)
        self._analysis_identity = identity

        file = self.backend.open()
        try:
            self.logger.debug("backend file: %s", file)
            self.logger.debug("backend file.keys: %s", list(file.keys()))
            self.logger.debug("backend file[%s].keys: %s", self.backend_name, list(file[self.backend_name].keys()))  # type: ignore
        except Exception as e:
            self.logger.debug("backend group %s not available yet: %s", self.backend_name, e)
        finally:
            file.close()

        self.log_prob = self.model.lnposterior_wbic if wbic else self.model.lnposterior

        self.sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim,
            log_prob_fn=self.log_prob,
            blobs_dtype=blobs_dtype,
            backend=self.backend,
            pool=pool,
            **self.kwargs,
        )

    def _current_analysis_identity(self):
        return fingerprint({
            "software": software_identity("emcee"), "model": self.model,
            "posterior": self.log_prob, "ndim": self.ndim, "nwalkers": self.nwalkers,
            "prior_names": self.model.prior_names, "wbic": self.wbic,
            "sampler_options": self.kwargs,
        })

    def _write_analysis_identity(self, identity):
        with self.backend.open("a") as handle:
            attrs = handle[self.backend_name].attrs
            attrs["jeanspy_analysis_identity"] = identity
            attrs["jeanspy_identity_format"] = IDENTITY_FORMAT
            handle[self.backend_name].create_dataset(
                "jeanspy_source_provenance", shape=(0,), maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

    def _record_source_provenance(self):
        record = source_provenance()
        with self.backend.open("a") as handle:
            group = handle[self.backend_name]
            history = group["jeanspy_source_provenance"]
            previous = json.loads(history.asstr()[-1]) if len(history) else None
            if previous is None or previous["source_sha256"] != record["source_sha256"]:
                record["iteration"] = int(group.attrs["iteration"])
                history.resize(len(history) + 1, axis=0)
                history[-1] = json.dumps(record, sort_keys=True)

    def _check_analysis_identity(self, identity):
        with self.backend.open("r") as handle:
            attrs = handle[self.backend_name].attrs
            stored = attrs.get("jeanspy_analysis_identity")
            version = attrs.get("jeanspy_identity_format")
        if stored != identity or version != IDENTITY_FORMAT:
            raise ValueError("Sampling analysis identity mismatch or missing legacy identity "
                             "(model, prior, data, schema, or solver). Use a new prefix or "
                             "explicitly reset to start a different analysis.")

    def check_parameter_conversion(self,p0_generator=None):
        """ check the conversion of parameters.
        This is useful to check if the parameters are properly converted from p0 to params.
        It will print the p0 and params and their comparison.
        """
        # Show some message to check the definition of the 'convert_params' 
        self.logger.info("Please check the following lines to see your parameters are properly converted:")
        if p0_generator is None:
            self.logger.info("p0_generator is None so we skip the conversion of p0. Please make sure to convert p0 properly by yourself.")
        else:
            p0 = p0_generator(None)
            params = self.model.convert_params(p0)
            self.logger.info("p0: %s", p0)
            self.logger.info("params:\n%s", params)
            # A model may combine sampled coordinates with additional fixed
            # physical parameters, so these two collections need not have the
            # same length. Log named sampling coordinates without aligning
            # unrelated rows of the physical parameter dictionary.
            coordinates = pd.Series(p0, index=getattr(self.model, "p_names_lnprob", None))
            self.logger.info("sampling coordinates:\n%s", coordinates)
    
    def set_wrapper_function(self):
        # NOTE: Here we define a global wrapper function for log_prob to accelerate the sampling.
        # Without the wrapper function, multiprocessing.pool will repeat the pickling/unpickling of the model
        # and it will be very slow.
        # Once we define the wrapper function, pool will only pickle/unpickle the wrapper function
        # and the model will be pickled only once. 
        # Note that the global variable/function will be copied to each worker process,
        # so it will not be shared among the workers, so it is safe to use.
        """Install the current log-probability callable in this worker process.

        Returns None. The module-level wrapper lets multiprocessing workers
        evaluate their initialized model without repeatedly serializing it.
        This is host-side worker initialization, not a numerical solver.
        """
        global log_prob_fn_wrapper
        def log_prob_fn_wrapper(p):
            """ wrapper function for log_prob to accelerate the sampling.
            """
            return self.log_prob(p)
        self.logger.info("log_prob_fn_wrapper defined.")


    def reset_pool(self,pool):
        """ reset the pool.
        """
        if self.pool is not None:
            raise RuntimeError("Sampler: pool is already set. Please reset the pool before setting a new one.")
        self.pool = pool
        self.sampler = emcee.EnsembleSampler(self.nwalkers,
                                             self.ndim,
                                             log_prob_fn=log_prob_fn_wrapper,
                                             blobs_dtype=self.sampler.blobs_dtype,
                                             backend=self.backend,
                                             pool=self.pool,
                                             **self.kwargs)


    @property
    def filename(self):
        """Return the HDF5 backend's filename."""
        return self.backend.filename
    

    def burn_in(self, nsteps, p0_generator, **kwargs):
        """Advance warmup and continue from its final ensemble.

        Warmup draws remain in the backend. Exclude them
        explicitly with get_chain(discard=...) when analyzing production draws.
        The chain already samples the posterior and must not be weighted by
        the posterior density a second time.
        """
        self._check_analysis_identity(self._current_analysis_identity())
        self.logger.info("Burn-in the sampler for %d steps.", nsteps)
        initial_state = None
        if self.backend.iteration == 0:
            self.check_parameter_conversion(p0_generator)
            # generate initial state
            initial_state = p0_generator(self.nwalkers)  # type: ignore
            # check if initial_state returns finite log_prob
            # if not, raise an error
            if not np.all(np.isfinite([self.model.lnposterior(p) for p in initial_state])):
                mes = []
                mes.append("Sampler: initial_state has non-finite log_prob")
                for p in initial_state:
                    if not np.all(np.isfinite(self.model.lnposterior(p))):
                        mes.append(f"p:{p}")
                        mes.append(f"lnposterior(p):{self.model.lnposterior(p)}")
                raise RuntimeError("\n".join(mes))
        self._record_source_provenance()
        self.sampler.run_mcmc(initial_state, nsteps,
                              progress=True,
                              **kwargs)
        self.logger.info("Burn-in completed.")
        return self.sampler.get_last_sample()


    def run_mcmc(self,
                 iterations,loops,
                 reset=False,
                 p0_generator: Union[Callable, None] = None,
                 enable_convergence_check=True,
                 **kwargs):
        """ run the sampler.
        p0 is generated by p0_generator.
        save and monitor the chain and lnprob using backends.
        blobs_dtype is obtained by model.

        iterations: number of iterations for each loop
        loops: number of loops
        reset: discard the stored chain and initialize a new run
        p0_generator: override the constructor's initial-state generator
        """
        # Set up the backend
        # Don't forget to clear it in case the file already exists

        self.logger.info("Running MCMC for %d iterations in %d loops.", iterations, loops)
        identity = self._current_analysis_identity()
        if not reset:
            self._check_analysis_identity(identity)

        if p0_generator is None:
            p0_generator = self.p0_generator
        if (reset or self.backend.iteration == 0) and p0_generator is None:
            raise ValueError("An initial-state generator is required for a new run.")
        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(loops)

        # This will be useful to testing convergence
        old_tau = np.inf

        initial_state = None
        if reset or self.backend.iteration == 0:
            self.check_parameter_conversion(p0_generator)
            # generate initial state
            initial_state = np.asarray(p0_generator(self.nwalkers), dtype=float)
            if initial_state.shape != (self.nwalkers, self.ndim):
                raise ValueError("Initial state must have shape (nwalkers, ndim).")
            if not np.isfinite(initial_state).all():
                raise ValueError("Initial state coordinates must be finite.")
            if not kwargs.get("skip_initial_state_check", False):
                if not emcee.ensemble.walkers_independent(initial_state):
                    raise ValueError("Initial state walkers must be linearly independent.")
            # check if initial_state returns finite log_prob
            # if not, raise an error
            if not np.all(np.isfinite([self.model.lnposterior(p) for p in initial_state])):
                mes = []
                mes.append("Sampler: initial_state has non-finite log_prob")
                for p in initial_state:
                    if not np.all(np.isfinite(self.model.lnposterior(p))):
                        mes.append(f"p:{p}")
                        mes.append(f"lnposterior(p):{self.model.lnposterior(p)}")
                raise RuntimeError("\n".join(mes))

        # Only discard persisted samples after the replacement has passed preflight.
        if reset:
            self.sampler.reset()
            self._write_analysis_identity(identity)
            self._analysis_identity = identity

        self._record_source_provenance()

        # Now we'll sample for up to  steps
        self.logger.info("iteration: %d", self.sampler.iteration)
        for i_loop in range(loops):
            # start mcmc sampling with self.sampler.run_mcmc and pool
            initial_state = None if self.backend.iteration > 0 else initial_state
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
                self.logger.info("Converged after %d iterations.", self.sampler.iteration)
                if enable_convergence_check:
                    break
                else:
                    self.logger.info("Converged but convergence_check is False, so continue sampling.")
            else:
                self.logger.info("Not converged yet.")
            self.logger.info("tau: %s", tau)
            self.logger.info("iteration: %d", self.sampler.iteration)
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
        if hasattr(self.model, "p_names_lnprob"):
            columns = self.model.p_names_lnprob
        else:
            columns = self.model.submodels["FlatPriorModel"].data.index.tolist()
        df = pd.DataFrame(chain,columns=columns)
        if with_lnprob:
            log_prob = self.backend.get_log_prob(flat=True,thin=thin,discard=discard)
            df["lnprob"] = log_prob
        return df
