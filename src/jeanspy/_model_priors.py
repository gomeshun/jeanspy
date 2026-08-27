"""Prior and fitting abstractions for the classical backend."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import cached_property

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._model_base import Model, logger


class FittableModel(Model,metaclass=ABCMeta):
    """ Abstract base class for fittable model.
    methods:
        load_data: load dataset required for parameter fitting or estimation.
        lnlikelihoods: define natural logarithm of the likelihood function.
        lnpriors: return a dictionary of natural logarithm of the prior probability of each parameter.
        lnposterior: return a dictionary of natural logarithm of the posterior probability of each parameter.
        convert_params: convert parameters from the parameter space to the model parameter space.
        prior_names: return a list of prior names.

    properties:
        inverse_temparature: inverse temparature of the model, given by 1/np.log(len(self.data))
        blobs_dtype: dictionary like {lnl: float, lnp1: float, lnp2: float, ...}. Where lnp1, lnp2, ... are obtained by self.lnpriors.keys().
    """

    def __init__(self,args_load_data=None,kwargs_load_data=None,*args,**kwargs):
        """ initialize FittableModel.

        Parameters
        ----------
        args_load_data: list
            arguments for load_data method.
        kwargs_load_data: dict
            keyword arguments for load_data method.
        args: list
            arguments for the parent class.
        kwargs: dict
            keyword arguments for the parent class.
        """
        super().__init__(*args,**kwargs)
        self.logger.info("Fittable Model: args_load_data: %r", args_load_data)
        # check if args_load_data is a list.
        if not isinstance(args_load_data,list):
            raise TypeError('args_load_data must be a list.')
        if kwargs_load_data is None:
            kwargs_load_data = {}
        self.logger.info("Fittable Model: kwargs_load_data: %r", kwargs_load_data)
        # check if kwargs_load_data is a dict.
        if not isinstance(kwargs_load_data,dict):
            raise TypeError('kwargs_load_data must be a dict.')
        self.load_data(*args_load_data,**kwargs_load_data)
        # check if self has the prior_names attribute.
        if not hasattr(self,'prior_names'):
            raise AttributeError('FittableModel must have the prior_names attribute.')


    @abstractmethod
    def convert_params(self,p):
        """ convert parameters from the parameter space to the model parameter space. """
        pass


    @abstractmethod
    def load_data(self,*args,**kwargs):
        """ load dataset required for parameter fitting or estimation.
        data must be stored in self.data, as a pd.DataFrame.
        additional data must be stored in self.additional_data, as a dict.
        """
        pass


    @cached_property
    def inverse_temparature(self,):
        """ inverse temparature of the model, given by 1/np.log(len(self.data))
        """
        n_data = self.n_data if hasattr(self,'n_data') else len(self.data)
        return 1/np.log(n_data)


    @abstractmethod
    def _lnlikelihoods(self,*args,**kwargs):
        """ define natural logarithm of the likelihood function.
        Here, any paramters must not given as args or kwargs, because they are internaly stored in self.params.
        """
        pass


    def lnlikelihoods(self,p,*args,**kwargs):
        """ calculate natural logarithm of the likelihood function.
        Note that this method changes the parameters of the model.
        p: ndarray: shape = (n_params,)
        """
        params = self.convert_params(p)
        self.update(params)
        lnl = self._lnlikelihoods(*args,**kwargs)
        return lnl


    def _lnlikelihood(self,*args,**kwargs):
        """ calculate natural logarithm of the likelihood function.
        Note that this method does not change the parameters of the model.
        """
        lnl = np.sum(self._lnlikelihoods(*args,**kwargs))
        if np.isnan(lnl):
            lnl = -np.inf
        return lnl

    def lnlikelihood(self,p,*args,**kwargs):
        """ calculate natural logarithm of the likelihood function.
        Note that this method changes the parameters of the model.
        p: ndarray: shape = (n_params,)
        """
        params = self.convert_params(p)
        self.update(params)
        lnl = self._lnlikelihood(*args,**kwargs)
        return lnl


    @abstractmethod
    def _lnpriors(self,p,*args,**kwargs):
        """ return a list of natural logarithm of the prior functions.
        Note that this method does not change the parameters of the model.
        The first argument is the parameter vector, p, before conversion.
        """
        pass


    def lnpriors(self,p,*args,**kwargs):
        """ return a dictionary of natural logarithm of the prior probability of each parameter.
        Note that this method changes the parameters of the model.
        p: ndarray: shape = (n_params,)
        """
        params = self.convert_params(p)
        self.update(params)
        lnp = self._lnpriors(*args,**kwargs)
        return lnp


    @property
    def blobs_dtype(self):
        return [ ("lnl",float), *[ (name, float) for name in self.prior_names ]]


    def lnposterior(self,p,*args,**kwargs):
        params = self.convert_params(p)
        self.update(params)
        lnl = -np.inf
        lnp_list = self._lnpriors(p,*args,**kwargs)
        if np.all([lnp > -np.inf for lnp in lnp_list]):
            lnl = self._lnlikelihood(*args,**kwargs)
        ret = (lnl + np.sum(lnp_list), lnl, *lnp_list)
        if np.isnan(ret[0]):
            mes = [
                f"lnposterior is nan. lnl:%s, lnp_list:%s" % (lnl, lnp_list),
                "p:%s" % p,
                "args:%s" % args,
                "kwargs:%s" % kwargs,
                "params:%s" % params
            ]
            self.logger.error("lnposterior is nan. lnl:%s, lnp_list:%s", lnl, lnp_list)
            self.logger.error("p:%s", p)
            self.logger.error("args:%s", args)
            self.logger.error("kwargs:%s", kwargs)
            self.logger.error("params:%s", params)
            raise ValueError(mes)
        return ret


    def lnposterior_wbic(self,p,*args,**kwargs):
        params = self.convert_params(p)
        self.update(params)
        lnl = -np.inf
        lnp_list = self._lnpriors(p,*args,**kwargs)
        if np.all([lnp > -np.inf for lnp in lnp_list]):
            lnl = self._lnlikelihood(*args,**kwargs) * self.inverse_temparature
        ret = (lnl + np.sum(lnp_list), lnl, *lnp_list)
        if np.isnan(ret[0]):
            mes = f"lnposterior_wbic is nan. lnl:{lnl}, lnp_list:{lnp_list}"
            mes += f"\np:{p}"
            mes += f"\nargs:{args}"
            mes += f"\nkwargs:{kwargs}"
            mes += f"\nparams:{params}"
            raise ValueError(mes)
        return ret


    @cached_property
    def ndim(self):
        return len(self.params_all)



class FlatPriorModel(Model):
    """ flat prior model.
    """
    required_param_names = []
    required_models = {}

    def __init__(self, config, show_init=False, submodels=None, **params):
        super().__init__(show_init, submodels or {}, **params)
        self.load_config(config)


    def load_config(self, config):
        """ load the upper and lower limits of each parameter from config.
        config: file name of the config file or pandas.DataFrame.
        NOTE: the upper and lower limits are for p (before conversion), not for params (after conversion).
        """
        if isinstance(config,str):
            try:
                self.fname_config = config
                self.data = pd.read_csv(config, index_col=0)
            except FileNotFoundError as e:
                logger.error("config file '%s' is not found.", config)
                raise(e)
        else:
            self.data = config

        self.lower = self.data["lower"].values
        self.upper = self.data["upper"].values

    def get_index(self,param_name):
        return self.data.index.get_loc(param_name)


    def extract_value_by_name(self,params,name):
        """get the value of the parameter from param (ndarray) based on the name of the parameter.

        Parameters
        ----------
        param: ndarray
            array of parameter values
        name: str
            name of the parameter

        Returns
        -------
        value: float
            value of the parameter
        """
        assert len(params) == len(self.data), f"len(param)={len(params)} != len(self.data)={len(self.data)}"
        return params[self.get_index(name)]


    def sample(self,size=None):
        size = (size,) if isinstance(size,int) else size
        size = size + (len(self.lower),) if isinstance(size,tuple) else size
        try:
            return np.random.uniform(self.lower, self.upper, size=size)
        except OverflowError as e:
            mes = f"OverflowError: lower:{self.lower}, upper:{self.upper}, size:{size}"
            e.args = (mes,) + e.args
            raise(e)


    def _lnprior(self,p):
        """ return a dictionary of natural logarithm of the prior probability of each parameter.

        if any member of p are out of the range defined in self.data, return -np.inf.
        if all members of p are in the range, return 0.
        Note that this method changes the parameters of the model.

        p: ndarray: shape = (n_params,)
        """
        lower = self.data["lower"].values
        upper = self.data["upper"].values
        if np.all((lower <= p) & (p <= upper)):
            return 0.0
        else:
            return -np.inf

    @staticmethod
    def generate_default_config_file(fname,param_names,lower=-np.inf,upper=np.inf):
        """ generate a default config file for FlatPriorModel.
        """
        df = pd.DataFrame({"lower":lower,"upper":upper},index=param_names)
        df.to_csv(fname)
        logger.info("generated %s.", fname)
        return df




class PhotometryPriorModel(Model):
    """ prior model for photometry.

    Parameters
    ----------
    loc : float
        Mean (location) of the log10 effective-radius prior in log10(pc).
        Pass ``float('nan')`` when the prior will be set later via
        :meth:`reset_prior`.
    scale : float
        Standard deviation (scale) of the log10 effective-radius prior.
        Pass ``float('nan')`` when the prior will be set later via
        :meth:`reset_prior`.
    """
    required_param_names = []
    required_models = {}

    def __init__(self, loc, scale, show_init=False, submodels=None, **params):
        super().__init__(show_init, submodels or {}, **params)
        print_dict = {"log10_re_pc": loc, "e_log10_re_pc": scale}
        self.logger.info("%s:%r", self.__class__.__name__, print_dict)
        self.reset_prior(loc, scale)

    def reset_prior(self,loc,scale):
        self._lnprior_func = norm(loc=loc,scale=scale).logpdf
        self._sample = norm(loc=loc,scale=scale).rvs


    def _lnprior(self,log10_re_pc):
        return self._lnprior_func(log10_re_pc)

    def sample(self,size):
        return self._sample(size=size)
