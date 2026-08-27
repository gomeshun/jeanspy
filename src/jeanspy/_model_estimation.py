"""Data handling and estimation models for the classical backend."""

from __future__ import annotations

import os
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._model_anisotropy import ConstantAnisotropyModel
from ._model_base import Model, logger
from ._model_jfactor import NFWModel
from ._model_priors import FittableModel, FlatPriorModel, PhotometryPriorModel
from ._model_profiles import PlummerModel
from ._model_solver import DSphModel


class DotDict(dict):
    """ Almost same as dict, but can be accessed by dot notation.
    """
    def __getattr__(self,key):
        # check if key is in self.
        if key in self:
            return self[key]
        else:
            super().__getattr__(key)

    def __setattr__(self,key,value):
        # check if key is in self.
        if key in self:
            self[key] = value
        else:
            super().__setattr__(key,value)

    def __delattr__(self,key):
        # check if key is in self.
        if key in self:
            del self[key]
        else:
            super().__delattr__(key)




class SimpleDSphEstimationModel(FittableModel,Model):
    """ A Simple model for dwarf spheroidal galaxy, considering only kinematical dataset.
    """

    required_param_names = []
    required_models = {
        "DSphModel": DSphModel,
        "FlatPriorModel": FlatPriorModel,
        "PhotometryPriorModel": PhotometryPriorModel,
    }

    dtype = np.float32

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        fname_config = self["FlatPriorModel"].fname_config
        self.logger.info("%s: Please check the consistensy of model parameters and config file: %s.", self.__class__, fname_config)
        self.logger.info("%s", "="*32)
        comparison = {
            "config": self.p_names_lnprob,
            "params": self.required_param_names_combined,
        }
        if len(comparison["config"]) != len(comparison["params"]):
            mes = f"Length of comparison['config'] and comparison['params'] are different."
            mes += f"config: {len(comparison['config'])}, params: {len(comparison['params'])}"
        try:
            self.logger.info("%s", pd.DataFrame(comparison))
            self.logger.info("%s", "="*32)
            # check if comparison["config"] is consistent with comparison["params"] by backward matching.
            consistencies = [ (param in p) for p,param in zip(comparison["config"],comparison["params"])]
            assert all(consistencies)  # NOTE: We can find substring in string by using "in" operator.
        except ValueError as e:
            self.logger.error("%r", comparison)
            raise(e)
        except AssertionError as e:
            self.logger.error("ERROR: config and params are not consistent.")
            self.logger.error("config file: %s", self["FlatPriorModel"].fname_config)
            self.logger.error("%r", comparison)
            self.logger.error("%r", consistencies)
            raise(e)


    @property
    def p_names_lnprob(self):
        """ return a list of parameter names used as an input of lnprob.
        """
        return self["FlatPriorModel"].data.index.tolist()


    def convert_params(self, p):
        """ convert parameters from p to params.
        Here, required_param_names_combined of this model is
            []
        """
        p_names = self.p_names_lnprob
        param_names = self.required_param_names_combined
        def convert_param(name,p):
            # zif "log10_" is in name by using a method of string
            if "log10_" in name:
                return 10**p
            elif "bfunc_" in name:
                # Inverse function of b -> log10(1-b)
                return 1 - 10**p
            else:
                return p
        d = { param_name:convert_param(p_name,p) for p_name,param_name,p in zip(p_names,param_names,p)}
        return pd.Series(d)


    def load_data(self, data, shared=False):
        """ Load dataset required for parameter fitting or estimation.

        Parameters
        ----------
        data : pandas.DataFrame
            Observed kinematic data. Must contain at least a ``vlos_kms``
            column. Pass a DataFrame directly; no internal database fetch is
            performed.
        shared : bool, optional
            Whether to store data in shared memory for multiprocessing.
        """
        self.shared = shared
        data = data.astype(self.dtype)
        self.reset_data(data)


    def reset_data(self, data):
        """Reset the observed data and refresh derived prior bounds.

        Parameters
        ----------
        data : pandas.DataFrame
            Observed kinematic data used by the model. The ``vlos_kms`` column
            is used to update the ``vmem_kms`` bounds in ``FlatPriorModel``.

        Returns
        -------
        None
        """
        self.data = data
        # override FlatPriorModel config by data
        # for vmem_kms
        before_lower = float(self["FlatPriorModel"].data.loc["vmem_kms", "lower"])
        before_upper = float(self["FlatPriorModel"].data.loc["vmem_kms", "upper"])
        self.logger.info(
            "%s: Override FlatPriorModel config by data for vmem_kms: before lower=%.6f upper=%.6f",
            self.__class__.__name__,
            before_lower,
            before_upper,
        )
        lower = self.data["vlos_kms"].min()
        upper = self.data["vlos_kms"].max()
        self["FlatPriorModel"].data.loc["vmem_kms","lower"] = lower
        self["FlatPriorModel"].data.loc["vmem_kms","upper"] = upper
        self.logger.info(
            "%s: Override FlatPriorModel config by data for vmem_kms: after lower=%.6f upper=%.6f",
            self.__class__.__name__,
            float(lower),
            float(upper),
        )


    @property
    def shared_memory_basename(self):
        if not self.shared:
            return None
        return f"SimpleDSphEstimationModel_{id(self)}"


    @property
    def data(self):
        if not self.shared:
            return self._data
        else:
            try:
                # If shared memory is already initialized, return the data.
                return DotDict({
                    "R_pc": np.ndarray(self.shared_shape,dtype=self.dtype,buffer=self.shm_R_pc.buf),
                    "vlos_kms": np.ndarray(self.shared_shape,dtype=self.dtype,buffer=self.shm_vlos_kms.buf),
                    "e_vlos_kms": np.ndarray(self.shared_shape,dtype=self.dtype,buffer=self.shm_e_vlos_kms.buf)
                })
            except FileNotFoundError as e:
                # if shared memory is not initialized, raise an error.
                self.logger.error("%s: SharedMemory '%s' is not initialized yet.", self.__class__.__name__, self.shared_memory_basename)
                raise(e)
            except AttributeError as e:
                # if shared memory is not initialized, raise an error.
                self.logger.error("%s: SharedMemory '%s' is not initialized yet.", self.__class__.__name__, self.shared_memory_basename)
                raise(e)



    @property
    def n_data(self):
        return self._n_data

    @data.setter
    def data(self,data: pd.DataFrame):
        self._n_data = len(data)
        if not self.shared:
            self._data = DotDict({
                "R_pc": data["R_pc"].values,
                "vlos_kms": data["vlos_kms"].values,
                "e_vlos_kms": data["e_vlos_kms"].values
            })
        else:
            self.logger.info("%s: Initialize shared memory '%s'", self.__class__.__name__, self.shared_memory_basename)
            self.shared_shape = data["R_pc"].shape
            assert self.shared_shape == data["vlos_kms"].shape
            assert self.shared_shape == data["e_vlos_kms"].shape
            self.buffer_size = data["R_pc"].values.nbytes
            assert self.buffer_size == data["vlos_kms"].values.nbytes
            assert self.buffer_size == data["e_vlos_kms"].values.nbytes
            # If shared memory is not initialized, initialize it.
            try:
                self.shm_R_pc = SharedMemory(name=self.shared_memory_basename+"_R_pc",create=True, size=self.buffer_size)
                R_pc = np.ndarray(self.shared_shape,dtype=self.dtype,buffer=self.shm_R_pc.buf)
                R_pc[:] = data["R_pc"].values
            except FileExistsError as e:
                self.shm_R_pc = SharedMemory(name=self.shared_memory_basename+"_R_pc",create=False)
                R_pc = np.ndarray(self.shared_shape,dtype=self.dtype,buffer=self.shm_R_pc.buf)
            try:
                self.shm_vlos_kms = SharedMemory(name=self.shared_memory_basename+"_vlos_kms",create=True, size=self.buffer_size)
                vlos_kms = np.ndarray(self.shared_shape,dtype=self.dtype,buffer=self.shm_vlos_kms.buf)
                vlos_kms[:] = data["vlos_kms"].values
            except FileExistsError as e:
                self.shm_vlos_kms = SharedMemory(name=self.shared_memory_basename+"_vlos_kms",create=False)
                vlos_kms = np.ndarray(self.shared_shape,dtype=self.dtype,buffer=self.shm_vlos_kms.buf)
            try:
                self.shm_e_vlos_kms = SharedMemory(name=self.shared_memory_basename+"_e_vlos_kms",create=True, size=self.buffer_size)
                e_vlos_kms = np.ndarray(self.shared_shape,dtype=self.dtype,buffer=self.shm_e_vlos_kms.buf)
                e_vlos_kms[:] = data["e_vlos_kms"].values
            except FileExistsError as e:
                self.shm_e_vlos_kms = SharedMemory(name=self.shared_memory_basename+"_e_vlos_kms",create=False)
                e_vlos_kms = np.ndarray(self.shared_shape,dtype=self.dtype,buffer=self.shm_e_vlos_kms.buf)
            data = {
                "R_pc": R_pc,
                "vlos_kms": vlos_kms,
                "e_vlos_kms": e_vlos_kms
            }
            data = DotDict(data)
            return data



    def _release_shared_memory(self,suffix):
        if self.shared:
            name = self.shared_memory_basename+suffix
            if not hasattr(self,"shared_shape"):
                raise ValueError(f"{self.__class__.__name__}: try to release shared memory {name} before initialization.")
            try:
                # access to the atrtibute of the instance self.shm{suffix}
                shm = getattr(self,f"shm{suffix}")
                shm.close()
                shm.unlink()  # raise FileNotFoundError if the shared memory is already unlinked.
                self.logger.info("%s: shared memory '%s' is released.", self.__class__.__name__, name)
                self.logger.info("id(self):%s", id(self))
            except FileNotFoundError as e:
                # alreadly unlinked. Do nothing.
                self.logger.info("%s: shared memory '%s' is already released.", self.__class__.__name__, name)
                self.logger.info("id(self):%s", id(self))


    def release_shared_memory(self):
        self._release_shared_memory("_R_pc")
        self._release_shared_memory("_vlos_kms")
        self._release_shared_memory("_e_vlos_kms")


    def _lnlikelihoods(self):
        """ define natural logarithm of the likelihood function. """
        s2 = self["DSphModel"].sigmalos2_dequad(self.data.R_pc)
        err2 = self.data.e_vlos_kms**2
        vmem_kms = self["DSphModel"].params.vmem_kms
        return norm.logpdf(self.data.vlos_kms,loc=vmem_kms,scale=np.sqrt(s2+err2))


    def _lnpriors(self,p_before_conversion):
        """ define natural logarithm of the prior function. """
        idx_log10_re_pc = self["FlatPriorModel"].get_index("log10_re_pc")
        log10_re_pc = p_before_conversion[idx_log10_re_pc]
        return [
            self["FlatPriorModel"]._lnprior(p_before_conversion),
            self["PhotometryPriorModel"]._lnprior(log10_re_pc)
            ]


    prior_names = ["flat_prior","photometry_prior"]


    def sample(self,size=None):
        """ sample from the model. """
        p = self["FlatPriorModel"].sample(size)
        # NOTE: p is ndarray with shape =
        #       - (n_params,) when size is None
        #       - (size, n_params) when size is int
        #       - (*size, n_params) when size is tuple
        # override log10_re_pc
        idx_log10_re_pc = self["FlatPriorModel"].get_index("log10_re_pc")
        p[ ..., idx_log10_re_pc ] = self["PhotometryPriorModel"].sample(size)
        return p


    def sample_data(self,size=None):
        """ sample data from the model. """
        s2 = self["DSphModel"].sigmalos2_dequad(self.data.R_pc)
        err2 = self.data.e_vlos_kms**2
        vmem_kms = self["DSphModel"].params.vmem_kms
        sampled_vlos_kms = norm.rvs(loc=vmem_kms,scale=np.sqrt(s2+err2),size=size)
        return sampled_vlos_kms



def get_default_estimation_model(data,
                                 photometry_prior_loc,
                                 photometry_prior_scale,
                                 config="priorconfig.csv"):
    """ Return a default estimation model.

    Parameters
    ----------
    data : pandas.DataFrame
        Observed kinematic data with at least ``vlos_kms``, ``R_pc``, and
        ``e_vlos_kms`` columns.
    photometry_prior_loc : float
        Mean of the log10 effective-radius prior (log10 pc).
    photometry_prior_scale : float
        Standard deviation of the log10 effective-radius prior.
    config : str, optional
        Path to the flat-prior configuration CSV file.  A default file is
        generated automatically when the path does not exist.
    """

    dsph_model = DSphModel(submodels={
        "StellarModel" : PlummerModel(),
        "DMModel" : NFWModel(),
        "AnisotropyModel" : ConstantAnisotropyModel(),
    })

    # Check if config file exists.
    if not os.path.exists(config):
        logger.warning("config file '%s' is not found.", config)
        logger.info("generate a default config file.")
        FlatPriorModel.generate_default_config_file(config,dsph_model.params_all.index)

    mdl = SimpleDSphEstimationModel(
        args_load_data=[data],
        submodels={
            "DSphModel" : DSphModel(submodels={
                "StellarModel" : PlummerModel(),
                "DMModel" : NFWModel(),
                "AnisotropyModel" : ConstantAnisotropyModel(),
            }),
            "FlatPriorModel": FlatPriorModel(config=config),
            "PhotometryPriorModel": PhotometryPriorModel(
                loc=photometry_prior_loc,
                scale=photometry_prior_scale,
            )
        })
    return mdl



class KI17_Model:
    def __init__(self,params_KI17_Model):
        """
        params_KI17_model: pandas.Series, index = (params_DSphModel,params_FG_model,s)
        """
        pass
