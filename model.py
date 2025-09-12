from __future__ import annotations
# from collections.abc import MutableMapping
from typing import Dict, Iterator, Any, Mapping, Optional
from collections.abc import MutableMapping

import pandas as pd
import numpy as np
import multiprocessing as multi
from multiprocessing.shared_memory import SharedMemory
from copy import copy
import os
from numpy import array,pi,sqrt,exp,power,log,log10,log1p,cos,tan,sin, sort,argsort, inf, isnan
from scipy.stats import norm
from scipy.special import k0, betainc, beta, hyp2f1, erf, gamma, gammainc
from scipy import integrate
from scipy.constants import parsec, degree, physical_constants # parsec in meter, degree in radian
from scipy.integrate import quad
from scipy.interpolate import interp1d,Akima1DInterpolator

from multiprocessing import Pool, cpu_count
from abc import ABCMeta, abstractmethod
from functools import cached_property, partial

import warnings

# logger
from logging import getLogger, StreamHandler, Formatter
# initialize logger
logger = getLogger(__name__)
handler = StreamHandler()
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.setLevel("INFO")

from .dequad import dequad
# check if dsph_database is installed
try:
    import dsph_database.spectroscopy
    import dsph_database.photometry
except ImportError:
    dsph_database = None
    warnings.warn("dsph_database is not installed. Some functionalities may not work.")

GMsun_m3s2 = 1.32712440018e20
R_trunc_pc = 1866.

kg_eV = 1./physical_constants["electron volt-kilogram relationship"][0]
im_eV = 1./physical_constants["electron volt-inverse meter relationship"][0]
solar_mass_kg = 1.9884e30
C0 = (solar_mass_kg*kg_eV)**2*((1./parsec)*im_eV)**5
C1 = (1e9)**2 * (1e2*im_eV)**5
C_J = C0/C1


class Parameters(MutableMapping):
    """
    Lightweight substitute for `pd.Series` used in Model.params.
    * Dot access           : p.re_pc
    * Dict compatibility    : p['re_pc']
    * update method support : p.update({...} or pd.Series)
    * Conversion with pandas: p.to_series()
    """

    __slots__ = ("_data",)

    # ---------- Basic ----------
    def __init__(self,
                 data: Optional[Mapping[str, Any]] = None,
                 **kw: Any) -> None:
        object.__setattr__(self, "_data", dict())
        if data is not None:
            if isinstance(data, pd.Series):
                object.__getattribute__(self, "_data").update(data.to_dict())
            elif isinstance(data, Parameters):
                object.__getattribute__(self, "_data").update(data._data)
            else:
                object.__getattribute__(self, "_data").update(data)
        if kw:
            object.__getattribute__(self, "_data").update(kw)

    # ---------- MutableMapping ----------
    def __getitem__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        return data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        data = object.__getattribute__(self, "_data")
        data[key] = value

    def __delitem__(self, key: str) -> None:
        data = object.__getattribute__(self, "_data")
        del data[key]

    def __iter__(self) -> Iterator[str]:
        data = object.__getattribute__(self, "_data")
        return iter(data)

    def __len__(self) -> int:
        data = object.__getattribute__(self, "_data")
        return len(data)

    # ---------- Python conventions ----------
    def __repr__(self) -> str:
        data = object.__getattribute__(self, "_data")
        kv = ", ".join(f"{k}={v!r}" for k, v in data.items())
        return f"Parameters({kv})"

    # ---------- Dot access ----------
    def __getattr__(self, name: str) -> Any:
        try:
            # Use object.__getattribute__ to avoid triggering __getattr__ recursively
            data = object.__getattribute__(self, "_data")
            return data[name]
        except (KeyError, AttributeError) as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        # _data is in __slots__, so set it directly
        if name == "_data":
            object.__setattr__(self, name, value)
        else:
            # Use object.__getattribute__ to avoid triggering __getattr__
            try:
                data = object.__getattribute__(self, "_data")
                data[name] = value
            except AttributeError:
                # _data hasn't been initialized yet
                object.__setattr__(self, "_data", {name: value})

    # ---------- Pickling support ----------
    def __getstate__(self):
        return {"_data": self._data}
    
    def __setstate__(self, state):
        self._data = state["_data"]

    # ---------- Compatibility utilities ----------
    def update(self, other: Mapping[str, Any] | "Parameters" | pd.Series,
               **kw: Any) -> None:
        """
        Accepts dict, Parameters, or pd.Series as input.
        Does not return a value (mimics pandas' surface API).
        """
        data = object.__getattribute__(self, "_data")
        if isinstance(other, pd.Series):
            data.update(other.to_dict())
        elif isinstance(other, Parameters):
            data.update(other._data)
        else:
            data.update(dict(other))
        if kw:
            data.update(kw)

    # Used for conversion with pandas (intended for internal use in Model)
    def to_series(self) -> pd.Series:
        data = object.__getattribute__(self, "_data")
        return pd.Series(data, name="params")

    # Series compatibility properties (minimum required)
    @property
    def index(self):
        data = object.__getattribute__(self, "_data")
        return list(data.keys())

    @property
    def values(self):
        data = object.__getattribute__(self, "_data")
        return list(data.values())
    
    def copy(self) -> "Parameters":
        """
        Return a shallow copy of the Parameters object.
        """
        data = object.__getattribute__(self, "_data")
        return Parameters(data)  
    
    def __deepcopy__(self, memo):
        """
        Return a deep copy of the Parameters object.
        """
        data = object.__getattribute__(self, "_data")
        return Parameters(copy.deepcopy(data, memo=memo))

class Model(metaclass=ABCMeta):
    #params, required_param_names = pd.Series(), ['',]
    '''base class of model objects.

    attributes:
        name: str, name of the model
        params: pd.Series, parameters of the model
        required_param_names: list of str, required parameters' names
        required_models: dict of {name: model_class}, required submodels' names and classes
        submodels: dict of {name: model_object}, submodels' names and objects
        
    
    methods:
        __init__(show_init=False, submodels=None, **params):
            Load parameters and check if all required parameters are given.
            Load submodels and check if all required models are given.
            Set model name as a combination submodels' names.
        __repr__():
            show model name and parameters.
        params_all():
            show all parameters as a pd.Series.
        required_param_names_combined():
            show required parameters' name recursively.
        is_required_param_names(param_names_candidates):
            check if param_names_candidates are in self.required_param_names
        update(new_params_dict=None,target='all',**kwargs):
            update model parameters recurrently.

    
    Note: "self.params", "self.required_models" and "self.required_param_names" are undefined.
        They must be defined in child class.
        
        self.required_models is a dict of {name: model_class}
        self.required_param_names is a list of str
    '''
    

    def __init__(self,show_init=False,submodels=None,**params):
        """
        Load parameters and check if all required parameters are given.
        Load submodels and check if all required models are given.
        Set model name as a combination submodels' names.

        Parameters
        ----------
        show_init: bool, if True, show parameters after initialization.
        submodels: dict of {name: model_object}, submodels' names and objects
        params: dict of {name: value}, parameters' names and values
        """
        self.name = self.__class__.__name__
        self.logger = logger.getChild(self.name)
        if submodels is None:
            submodels = {}

        # check if the model has "required_param_names" and "params" attributes.
        if not hasattr(self,'required_param_names'):
            raise AttributeError(self.name+' has no attribute "required_param_names"')
        
        # check if the model has "required_models" attribute.
        if not hasattr(self,'required_models'):
            raise AttributeError(self.name+' has no attribute "required_models"')
        # check if all required models are given.
        if set(self.required_models.keys()) != set(submodels.keys()):
            raise ValueError(self.name+' has the models: '+str(self.required_models.keys())+" but input is "+str(submodels.keys()))
        else:
            # load submodels
            self.submodels = pd.Series(submodels)

        # initialize parameters of this model
        self.params = Parameters({ p:np.nan for p in self.required_param_names})
        self._parammap: Dict[str, "Model"] = {}
        self._build_parammap()
        self.update(params,target="all")

        # # check if all required parameters are given.
        # if set(params.keys()) != set(self.required_param_names):
        #     raise ValueError(self.name+' has the paramsters: '+str(self.required_param_names)+" but input is "+str(params.keys()))
        
        # set model name
        if len(self.submodels) > 0:
            self.name += "_" + '+'.join((model.name for model in self.submodels.values))

        # check the consistency of params_all and required_param_names_combined
        params_all_index = self.params_all.index  #[ pname.split(":")[-1] for pname in self.params_all.index]
        required_param_names_combined = self.required_param_names_combined 
        if not np.all(params_all_index == required_param_names_combined):
            raise ValueError("params_all and required_param_names_combined are inconsistent: "+str(params_all_index)+" vs "+str(required_param_names_combined))
            
        if show_init:
            self.logger.info("initialized:\n%s", self)
    

    def _as_dataframe(self):                           # ← 上書き
        """ Convert the model parameters to a DataFrame for better readability.
        """
        # 1) (<path>, <param>) → value の辞書を作る --------------------
        tuples = []
        values = []
        for full_key, val in self.params_all_with_model_name.items():
            # full_key = "StellarModel:re_pc" など
            if ":" in full_key:
                path, param = full_key.split(":", 1)
            else:
                path, param = self.__class__.__name__, full_key
            tuples.append((path, param))
            values.append(val)

        # 2) MultiIndex DataFrame に変換 ------------------------------
        idx = pd.MultiIndex.from_tuples(tuples, names=["model", "param"])
        df = pd.DataFrame({"value": values}, index=idx)

        # 3) 見やすさ調整（Optional）----------------------------------
        with pd.option_context("display.max_rows", None,
                            "display.max_colwidth", 20,
                            "display.precision", 6):
            return df
        

    def __repr__(self):
        return self._as_dataframe().to_string()  # __str__() を上書きしているので、print() で表示される  


    def __str__(self):
        return self._as_dataframe().to_string()
        

    def __getitem__(self,key):
        """ syntax sugar for self.submodels[key] """
        return self.submodels[key]


    def _repr_html_(self):
        return self._as_dataframe().to_html()  # __str__() を上書きしているので、print() で表示される
    

    def _build_parammap(self):
        """ build a map of parameters and models.
        """
        for p in self.required_param_names:
            self._parammap[p] = self
        for mdl in self.submodels.values:
            mdl._build_parammap()
            self._parammap.update(mdl._parammap)


    @property
    def params_all(self):
        """ show all parameters as a pd.Series.
        """
        merged = Parameters(self.params)
        for mdl in self.submodels.values:
            merged.update(mdl.params_all)
        return merged

    @property
    def params_all_with_model_name(self):
        """ show all parameters as a pd.Series.
        """
        merged = Parameters()
        merged.update(self.params)        # 自分はそのまま

        for name, mdl in self.submodels.items():
            tmp = Parameters({f"{name}:{k}": v for k, v in mdl.params_all_with_model_name.items()})
            merged.update(tmp)

        return merged

    

    @property
    def required_param_names_combined(self):
        """ show all required parameters' name recursively.
        """
        # load required parameters' name of this model
        ret = self.required_param_names[:] # need copy because we must keep self.required_param_names
        if len(self.submodels) > 0: # if there are submodels
            # add submodels' required parameters' name
            [ ret.extend(model.required_param_names_combined) for model in self.submodels.values ]
        return ret


    def is_required_param_names(self,param_names_candidates):
        return [ (p in self.required_param_names) for p in param_names_candidates ]


    def update(self,
            new_params=None,
            target: str = "all",  # No target means all, just for compatibility with other models
            **kwargs):
        merged: Dict[str, Any] = {}

        if new_params is not None:
            if isinstance(new_params, Parameters):
                merged.update(new_params._data)
            elif isinstance(new_params, pd.Series):
                merged.update(new_params.to_dict())
            else:
                merged.update(dict(new_params))
        if kwargs:
            merged.update(kwargs)
        for key, val in merged.items():
            try:
                owner = self._parammap[key]
            except KeyError:
                raise ValueError(f"Unknown parameter '{key}' for model '{self.name}'.")
            owner.params[key] = val

class StellarModel(Model):
    """Base class of StellarModel objects.
    """
    name = "stellar Model"
    required_models = {}

    def density(self,distance_from_center,dimension):
        if dimension == "2d":
            return self.density_2d(distance_from_center)
        elif dimension == "3d":
            return self.density_3d(distance_from_center)
    def density_2d_truncated(self,R_pc,R_trunc_pc):
        """
        Truncated 2D density. Note that
            \int_0^{R_trunc} 2\pi R density_2d_truncated(R,R_trunc) = 1 .
        """
        return self.density_2d(R_pc)/self.cdf_R(R_trunc_pc)
    
    @abstractmethod
    def density_2d(self,R_pc):
        pass
    
    @abstractmethod
    def density_3d(self,r_pc):
        pass

    

class PlummerModel(StellarModel):
    name = "Plummer Model"
    required_param_names = ['re_pc',]
    required_models = {}
    
    
    def density_2d(self,R_pc):
        re_pc= self.params.re_pc
        return 1/(1+(R_pc/re_pc)**2)**2 /np.pi/re_pc**2
    
    
    def logdensity_2d(self,R_pc):
        re_pc= self.params.re_pc
        return -np.log1p((R_pc/re_pc)**2)*2 -log(np.pi) -log(re_pc)*2
    
    def density_2d_normalized_re(self,R_pc):
        re_pc= self.params.re_pc
        return 4/(1+(R_pc/re_pc)**2)**2
      
    def density_3d(self,r_pc):
        re_pc= self.params.re_pc
        return (3/4/np.pi/re_pc**3)/np.sqrt(1+(r_pc/re_pc)**2)**5
    
    
    def cdf_R(self,R_pc):
        '''
        cdf_R(R) = \int_0^R \dd{R'} 2\pi R' \Sigma(R')
        '''
        re_pc= self.params.re_pc
        return 1/(1+(re_pc/R_pc)**2)
    
    def mean_density_2d(self,R_pc):
        '''
        return the mean density_2d in R < R_pc with the weight 2*pi*R
        mean_density_2d = \frac{\int_\RoIR \dd{R} 2\pi R \Sigma(R)}{\int_\RoIR \dd{R} 2\pi R}
            = \frac{cdf_R(R)}{\pi R^2}
        '''
        re_pc= self.params.re_pc
        return 1/pi/(R_pc**2+re_pc**2)
    
    def _half_light_radius(self,re_pc):
        '''
        Half-light-raduis means that the radius in which the half of all stars are include
        '''
        return re_pc
      
    def half_light_radius(self):
        '''
        Half-light-raduis means that the radius in which the half of all stars are include
        '''
        return self._half_light_radius(self.params.re_pc)

    
    
class SersicModel(StellarModel):
    name = "SersicModel"
    required_param_names = ['re_pc','n']
    required_models = {}
    
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        dirname = os.path.dirname(__file__)
        df = pd.read_csv(f"{dirname}/sersic_log10n_log10bn.csv")
        self._b_interp = interp1d(df["log10n"].values,df["log10bn"].values,"cubic",assume_sorted=True)
        self.coeff = pd.read_csv(f"{dirname}/coeff_dens_mod.csv",comment="#",delim_whitespace=True,header=None).values
    
    @property
    def b_approx(self):
        n = self.params.n
        return 2*n - 0.324
    
    @property
    def b_CB(self):
        # approximation by Eq.(18) of Ciotti and Bertin (1999), [arXiv:astro-ph/9911078]
        # It is valid for n > 0.5.
        n = self.params.n
        return 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2) + 131/(1148175*n**3) - 2194697/(
 30690717750*n**4)
    
    @property
    def b(self):
        n = self.params.n
        return 10**self._b_interp(log10(n))
    
    @property
    def norm(self):
        n = self.params.n
        return pi*self.params.re_pc**2 *power(self.b,-2*n) * gamma(2*n+1)
    
    def density_2d(self,R_pc):
        re_pc= self.params.re_pc
        n = self.params.n
        return exp(-self.b*power(R_pc/self.params.re_pc,1/n))/self.norm
    
    def density_2d_normalized_re(self,R_pc):
        re_pc= self.params.re_pc
        n = self.params.n
        return exp(-self.b*(power(R_pc/self.params.re_pc,1/n)-1))
    
    
    def cdf_R(self,R_pc):
        '''
        cdf_R(R) = \int_0^R \dd{R'} 2\pi R' \Sigma(R')
        '''
        re_pc= self.params.re_pc
        n = self.params.n
        return gammainc(2*n,self.b*power(R_pc/re_pc,1/n)) # - gammainc(2*n,0)
        
    def mean_density_2d(self,R_pc):
        '''
        return the mean density_2d in R < R_pc with the weight 2*pi*R
        mean_density_2d = \frac{\int_\RoIR \dd{R} 2\pi R \Sigma(R)}{\int_\RoIR \dd{R} 2\pi R}
            = \frac{cdf_R(R)}{\pi R^2}
        '''
        return self.cdf_R(R_pc)/pi/R_pc**2
    
    @property
    def p_LGM(self):
        n = self.params.n
        return 1 - 0.6097/n + 0.05463/n**2
    
    @property
    def norm_3d(self):
        Rhalf = self.params.re_pc
        n = self.params.n
        b = self.b_CB
        p = self.p_LGM
        ind = (3-p)*n
        return 4 * pi * Rhalf**3 * n * gamma(ind) / b**ind
    
    def density_3d_LGM(self,r_pc):
        p = self.p_LGM
        n = self.params.n
        b = self.b_CB
        x = (r_pc/self.params.re_pc)
        return x**-p * exp(-b * x**(1/n)) / self.norm_3d
    
    def density_3d(self,r_pc):
        pass
    
    def half_light_radius(self):
        return self.params.re_pc
    
    
    
class Exp2dModel(StellarModel):
    """Stellar model whose 2D (projected, surface) density is given by the exponential model.
    """
    name = "Exp2dModel"
    required_param_names = ['re_pc',]
    required_models = {}
    
    
    @property
    def R_exp_pc(self):
        return self.params.re_pc/1.67834699001666
    
    def density_2d(self,R_pc):
        re_pc = self.R_exp_pc
        return (1./2/pi/re_pc**2)*exp(-R_pc/re_pc) 
    
    def logdensity_2d(self,R_pc):
        re_pc = self.R_exp_pc
        return log(1./2/pi) -log(re_pc)*2 +(-R_pc/re_pc) 
    
    def density_3d(self,r_pc):
        re_pc = self.R_exp_pc
        return (1./2/pi**2/re_pc**3)*k0(r_pc/re_pc)
    
    def cdf_R(self,R_pc):
        '''
        cdf_R(R) = \int_0^R \dd{R'} 2\pi R' \Sigma(R')
        '''
        re_pc = self.R_exp_pc
        return 1. - exp(-R_pc/re_pc)*(1+R_pc/re_pc)
    
    def mean_density_2d(self,R_pc):
        re_pc = self.R_exp_pc
        return self.cdf_R(R_pc)/pi/R_pc**2
    
    def _half_light_radius(self,re_pc):
        return 1.67834699001666*self.R_exp_pc
    

    
    def half_light_radius(self):
        return self._half_light_radius(self.params.re_pc)
    
    
    
class Exp3dModel(StellarModel):
    """Stellar model whose 3D (deprojected) density is given by the exponential model.
    """
    name = "Exp3dModel"
    required_param_names = ['re_pc',]
    required_models = {}
    
    def density_2d(self,R_pc):
        re_pc = self.params.re_pc
        return (1./2/pi/re_pc**2)*exp(-R_pc/re_pc) 
    def density_3d(self,r_pc):
        re_pc = self.params.re_pc
        return (1./2/pi**2/re_pc**3)*k0(r_pc/re_pc)
    def cdf_R(self,R_pc):
        '''
        cdf_R(R) = \int_0^R \dd{R'} 2\pi R' \Sigma(R')
        '''
        re_pc = self.params.re_pc
        return 1. - exp(-R_pc/re_pc)*(1+R_pc/re_pc)
    def mean_density_2d(self,R_pc):
        re_pc = self.params.re_pc
        return self.cdf_R(R_pc)/pi/R_pc**2
    def half_light_radius(self):
        return 1.67834699001666*self.params.re_pc
        
        
        
class Uniform2dModel(StellarModel):
    name = "uniform Model"
    required_param_names = ['Rmax_pc',]
    required_models = {}
    
    def density_2d(self,R_pc):
        return 1./(pi*self.params.Rmax_pc**2)*np.ones_like(R_pc)
    def cdf_R(self,R_pc):
        return (R_pc/self.params.Rmax_pc)**2    

    
    
class DMModel(Model):
    name = "DM Model"
    
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.roi_deg_max_warning = 1.0  # maximum angle for evaluating J-factor
    
    @abstractmethod
    def mass_density_3d(self,r_pc):
        pass
    
    def assert_roi_is_enough_small(self,roi_deg):
        assert np.all(roi_deg<=self.roi_deg_max_warning)
    
    def jfactor_ullio2016_simple(self, dist_pc, roi_deg=0.5):
        """Calculate J-factor of DM profile using Eq.(B.10) in [arXiv:1603.07721].
        
        NOTE: The upper limit of the integral domain of Eq.(B.10) is \mathcal{R} (truncation radius),
        but it must be typo of R_\mathrm{max} (ROI). 
        Practically, the uper limit is max(R_\mathrm{max}, \mathcal{R}).
        """
        self.assert_roi_is_enough_small(roi_deg)
        roi_pc = dist_pc * np.sin(np.deg2rad(roi_deg))
        func = lambda r: r**2 * self.mass_density_3d(r)**2
        integ = dequad(func,0,roi_pc)
        j = 4 * np.pi / dist_pc**2 * integ * C_J
        return j
    
    def jfactor_ullio2016(self, dist_pc, roi_deg=0.5):
        """Calculate J-factor of DM profile using Eq.(B.10) in [arXiv:1603.07721].
        
        NOTE: The upper limit of the integral domain of Eq.(B.10) is \mathcal{R} (truncation radius),
        but it must be typo of R_\mathrm{max} (ROI). 
        Practically, the uper limit is max(R_\mathrm{max}, \mathcal{R}).
        """
        pass
        

class ZhaoModel(DMModel):
    name = "Zhao Model"
    required_param_names = ['rs_pc','rhos_Msunpc3','a','b','g','r_t_pc']
    required_models = {}
    
    
    def mass_density_3d(self,r_pc):
        rs_pc, rhos_Msunpc3,a,b,g = self.params.rs_pc, self.params.rhos_Msunpc3, self.params.a, self.params.b,self.params.g
        x = r_pc/rs_pc
        return rhos_Msunpc3*power(x,-g)*power(1+power(x,a),-(b-g)/a)
        
    def enclosure_mass(self,r_pc):
        rs_pc, rhos_Msunpc3,a,b,g = self.params.rs_pc, self.params.rhos_Msunpc3, self.params.a, self.params.b,self.params.g
        r_t_pc = self.params.r_t_pc
        
        # truncation
        # r_pc_trunc = copy(r_pc)
        # larger_than_r_t = r_pc > r_t_pc
        # r_pc_trunc = np.broadcast_to(r_pc_trunc,larger_than_r_t.shape)
        # r_pc_trunc = np.array(r_pc_trunc)    
        # r_pc_trunc[larger_than_r_t] = r_t_pc
        r_pc_trunc = np.where(r_pc>r_t_pc,r_t_pc,r_pc)
        
        x = power(r_pc_trunc/rs_pc,a)
        argbeta0 = (3-g)/a
        argbeta1 = (b-3)/a
        
        return (4.*pi*rs_pc**3 * rhos_Msunpc3/a) * beta(argbeta0,argbeta1) * betainc(argbeta0,argbeta1,x/(1+x))
        
        
class NFWModel(DMModel):
    name = "NFW Model"
    required_param_names = ['rs_pc','rhos_Msunpc3','r_t_pc']
    required_models = {}
    
    
    def mass_density_3d(self,r_pc):
        rs_pc, rhos_Msunpc3 = self.params.rs_pc, self.params.rhos_Msunpc3
        x = r_pc/rs_pc
        return rhos_Msunpc3/x/(1+x)**2
        
    def enclosure_mass(self,r_pc):
        threshold = 1e-7  # threshold to avoid underflow
        rs_pc, rhos_Msunpc3 = self.params.rs_pc, self.params.rhos_Msunpc3
        r_t_pc = self.params.r_t_pc
        # truncation
        if isinstance(r_pc,np.ndarray):
            # r_pc_trunc = copy(r_pc)
            # larger_than_r_t = r_pc > r_t_pc
            # r_pc_trunc = np.broadcast_to(r_pc_trunc,larger_than_r_t.shape)
            # r_pc_trunc = np.array(r_pc_trunc)
            # r_pc_trunc[larger_than_r_t] = r_t_pc
            r_pc_trunc = np.where(r_pc>r_t_pc,r_t_pc,r_pc)
        else:
            r_pc_trunc = min(r_pc,r_t_pc)
        x = r_pc_trunc/rs_pc
        ret = np.zeros_like(x)
        is_small = x < threshold  # DEBUG: 2021/10/28
        # NOTE: (1/(1+x)-1 + log(1+x)) = B(2,0,x/(1+x)), 
        # but scipy.special.betainc and scipy.special.beta are useless because of their diversence.
        # Therefore we use another expression in the following calculation.
        # Note that the element specification is relatively slow, thus we calculate all elements first and then modify overflowed ones.
        ret = (1/(1+x)-1 + log(1+x))  # NOTE:  underflow occurs when x<<1. 
        ret = np.array(ret)
        ret[is_small] = x[is_small]**2/2  # Series expantion of (1/(1+x)-1 + log(1+x)) up to the second order
        return (4.*pi*rs_pc**3 * rhos_Msunpc3) * ret 
    
    def jfactor_ullio2016_simple(self,dist_pc,roi_deg=0.5):
        self.assert_roi_is_enough_small(roi_deg)
        roi_pc = dist_pc*np.sin(np.deg2rad(roi_deg))
        rs_pc, rhos_Msunpc3 = self.params.rs_pc, self.params.rhos_Msunpc3
        r_t_pc = self.params.r_t_pc
        r_max_pc = np.min([roi_pc*np.ones_like(r_t_pc),r_t_pc],axis=0)
        c_max = r_max_pc/rs_pc
        j = C_J * 4 * pi * rs_pc**3 * rhos_Msunpc3**2 / dist_pc**2  # normalization
        j *= (1-1/(1+c_max)**3)/3 + ((rs_pc/dist_pc)**2 * c_max**3/(1+c_max)**3)/9  # approximation of W(r,0,r) upto second leading order
        return j
    
    def jfactor_evans2016(self,dist_pc,roi_deg=0.5):
        """J-factor fitting function given by https://arxiv.org/pdf/1604.05599.pdf
        Note that this formula causes the cancelation of significant digits
        """
        self.assert_roi_is_enough_small(roi_deg)
        def func_x(s):
            epsilon = 1e-8
            #print(f"s:{s.values}")
            s = np.atleast_1d(s)
            assert np.all(s>=0)
            ret = np.nan * np.ones_like(s)
            cond_1 = s<1-epsilon
            cond_2 = 1+epsilon<s
            cond_3 = np.abs(1-s) <= epsilon
            ret[cond_1] = np.arccosh(1/s[cond_1])/np.sqrt(1-s[cond_1]**2)
            ret[cond_2] = np.arccos(1/s[cond_2])/np.sqrt(s[cond_2]**2-1)
            ret[cond_3] = 1 - (2*(s[cond_3]-1))/3 + 7/15 * (s[cond_3]-1)**2
            #print(ret)
            return ret
        
        #func_x = lambda s : (
        #    1/np.arccosh(s)/np.sqrt(1-s**2) if 0<=s<=1 else 1/np.arccos(s)/np.sqrt(s**2-1)
        #)
        roi_pc = dist_pc * np.deg2rad(roi_deg)
        rs_pc, rhos_Msunpc3 = self.params.rs_pc, self.params.rhos_Msunpc3
        r_t_pc = self.params.r_t_pc
        r_max_pc = np.min([np.ones_like(r_t_pc)*roi_pc,r_t_pc],axis=0)
        y =  r_max_pc / rs_pc
        delta = 1 - y**2
        coeff_evans = (2*y*(7*y-4*y**3+3*pi*delta**2) + 6*(2*delta**3-2*delta-y**4)*func_x(y)) / 6 / delta**2
        epsilon = 1e-8
        coeff_evans[np.abs(1-y)<epsilon] = (np.pi - 38/15) + (-(64/21) + np.pi)*(y-1)
        j = C_J * 2 * pi * rhos_Msunpc3**2 * rs_pc**3 / dist_pc**2 * coeff_evans
        return j#np.log10(j)
        


class AnisotropyModel(Model):
    name = "AnisotropyModel"

    @abstractmethod
    def beta(self,r):
        pass
    

    @abstractmethod
    def f(self,r):
        pass


    @abstractmethod
    def kernel(self,u,R,**kwargs):
        """ Integrand of the LOSVD kernel function K(u).
        
        Parameters
        ----------
        u: float, 1d array
            u = r/R
        R: float, 1d array
            R: projected radius
        """
        pass

    
class ConstantAnisotropyModel(AnisotropyModel):
    name = "ConstantAnisotropyModel"
    required_param_names = ['beta_ani']
    required_models = {}
    
    def beta(self,r):
        return self.params.beta_ani
    

    def f(self,r):
        beta_ani = self.params.beta_ani
        return r ** (2*beta_ani)
    
    
    def kernel(self,u,R,**kwargs):
        """
        kernel function K(u). LOSVD is given by
        
            sigmalos2(R) = 2 * \int_1^\infty du \nu_\ast(uR)/\Sigma_\ast(R) * GM(uR) * K(u)/u.
        """
        b = self.params.beta_ani
        u2 = u**2
        kernel = sqrt(1-1/u2)*((1.5-b)*u2*hyp2f1(1.0,1.5-b,1.5,1-u2)-0.5)
        return kernel
    


class OsipkovMerrittModel(AnisotropyModel):
    name = "OsipkovMerrittModel"
    required_param_names = ["r_a"]
    required_models = {}
    

    def beta(self, r):
        r_a = self.params.r_a
        return r**2/(r**2+r_a**2)
    

    def f(self,r):
        r_a = self.params.r_a
        return (r_a**2+r**2)/r_a**2

    
    def kernel(self,u,R,**kwargs):
        """
        u, R: 1d array
        """
        # R = R[:,np.newaxis]  # axis = 0
        # u = u[np.newaxis, :] # axis = 1
        u_a = self.params.r_a / R
        u2_a = u_a**2
        u2 = u**2
        return (u2+u2_a)*(u2_a+0.5)/(u*(u2_a+1)**1.5) * np.arctan(np.sqrt((u2-1)/(u2_a+1))) - np.sqrt(1-1/u2)/2/(u2_a+1)

    
class BaesAnisotropyModel(AnisotropyModel):
    name = "BaesAnisotropyModel"
    required_param_names = ["beta_0", "beta_inf","r_a","eta"]
    required_models = {}
    
    
    def beta(self,r):
        b0,binf = self.params.beta_0, self.params.beta_inf
        r_a, eta = self.params.r_a, self.params.eta
        x = power(r/r_a,eta)
        return (b0+binf*x)/(1+x)
    
    def f(self,r):
        b0,binf = self.params.beta_0, self.params.beta_inf
        r_a, eta = self.params.r_a, self.params.eta
        x = power(r/r_a,eta)
        return power(r,2*b0)*power(1+x,2*(binf-b0)/eta)
        
    
    def integrand_kernel(self,u_integ,R):
        """
        u = r/R,
        us = r_a/R
        """
        u2_integ = u_integ**2
        r_integ = R*u_integ
        return u_integ/sqrt(u2_integ-1)*(1-self.beta(r_integ)/u2_integ)/self.f(r_integ)
    
    
    def kernel(self,u,R,**kwargs):
        """
        kernel function K(u). LOSVD is given by
        
            sigmalos2(R) = 2 * \int_1^\infty du \nu_\ast(uR)/\Sigma_\ast(R) * GM(uR) * K(u)/u.
            
        # u: ndarray, shape = (n_u,)
        # R: ndarray, shape = (n_R,)
        
        return: ndarray, shape = (n_R,n_u)
        """
        n = 128 if ("n" not in kwargs) else kwargs["n"]
        
        u_expanded = u[np.newaxis,:,np.newaxis]  # axis = 1
        R_expanded = R[:,np.newaxis,np.newaxis]  # axis = 0
        #print("u_shape:{} R.shape:{}".format(u.shape,R.shape))
        
        def integrand(_u):     
            #_u_array = np.array(_u)[np.newaxis,np.newaxis,:]  # axis = 2
            return self.integrand_kernel(_u,R_expanded)
            
        integration = dequad(integrand,1,u_expanded,n,axis=2,replace_inf_to_zero=True,replace_nan_to_zero=True)  # shape = (n_R, n_u)
            
        return integration * self.f(R_expanded[...,0]*u_expanded[...,0])/u_expanded[...,0]

    


class DSphModel(Model):
    name = 'DSphModel'
    required_param_names = ["vmem_kms"]
    required_models = {
        "StellarModel": StellarModel,
        "DMModel": DMModel,
        "AnisotropyModel": AnisotropyModel
    }
    ncpu = multi.cpu_count()
#    def __init__(self,StellarModel,DMModel,**params_DSphModel):
#        """
#        params_DSphModel: pandas.Series, index = (params_StellarModel,params_DMModel,center_of_dSph)
#        """
#        # NOTE IT IS NOT COM{PATIBLE TO THE CODE BELOW!!!
#        super().__init__(**params_DSphModel)
#        self.submodels = (StellarModel,DMModel)
#        self.name = ' and '.join((model.name for model in self.submodels))


    def _sigmar2(self,r_pc):
        RELERROR_INTEG = 1e-6
        density_3d = self["StellarModel"].density_3d
        enclosure_mass = self["DMModel"].enclosure_mass
        f = self["AnisotropyModel"].f
        integrand = lambda r: density_3d(r)*f(r)*GMsun_m3s2*enclosure_mass(r)/r**2/f(r_pc)/density_3d(r_pc)*1e-6/parsec
        integ, abserr = integrate.quad(integrand,r_pc,np.inf)
        return integ
    

    def sigmar2(self,r_pc):
        ''' Return the radial velocity dispersion squared at r_pc. '''
        return np.vectorize(self._sigmar2)(r_pc)
    

    def sigmat2(self,r_pc):
        ''' Return the tangential velocity dispersion squared at r_pc. '''
        beta = self["AnisotropyModel"].beta(r_pc)
        sigmar2 = self.sigmar2(r_pc)
        return sigmar2*(1-beta)


    
    def integrand_sigmalos2(self,u,R_pc,n_kernel=128):
        '''
        integrand of sigmalos2 at R = R_pc.
        u is a variable of integration, u=r/R.
        Domain: 1 < u < oo.
        
        u: ndarray: shape = (n_u,)
        R_pc: ndarray: shape = (n_R,)
        '''
        
        R_pc = np.array(R_pc)[:,np.newaxis] # axis = 0
        u = np.array(u)[np.newaxis,:]  # axis = 1
        
        density_3d = self["StellarModel"].density_3d
        density_2d = self["StellarModel"].density_2d
        enclosure_mass = self["DMModel"].enclosure_mass
        kernel = self["AnisotropyModel"].kernel
        r = R_pc*u
        # Note that parsec = parsec/m.
        # If you convert m -> pc,      ... var[m] * [1 pc/ parsec m] = var/parsec[pc].
        #                pc^1 -> m^pc, ... var[pc^1] * parsec(=[pc/m]) = var[m^-1]
        # Here var[m^3 pc^-1 s^-2] /parsec[m/pc] * 1e-6[km^2/m^2] = var[km^2/s^2]
        return 2.0 * kernel(u,R_pc,n=n_kernel)/u *  density_3d(r)/density_2d(R_pc)*GMsun_m3s2 * enclosure_mass(r) / parsec * 1e-6

    
    def sigmalos2_dequad(self,R_pc,n=1024,n_kernel=128,ignore_RuntimeWarning=True):
        def func(u):
            '''
            shape: (n_u,) -> (n_R,n_u)
            Note that the shape of kernel return. 
            '''
            return self.integrand_sigmalos2(u,R_pc,n_kernel)
        with warnings.catch_warnings():
            if ignore_RuntimeWarning:
                warnings.simplefilter('ignore',RuntimeWarning)
            integ = dequad(func,1,np.inf,axis=-1,n=n,replace_inf_to_zero=True,replace_nan_to_zero=True)
            # sanity check: sigmalos2 should be positive.
            # If not, raise ValueError with the value of R_pc and sigmalos2
            # and with current model parameters.
            if np.any(integ<0):
                errmes = "sigmalos2 is negative at R_pc = {} pc.".format(R_pc[integ<0])
                errmes += "with sigmalos2 = {}".format(integ[integ<0])
                errmes += "with current model parameters: {}".format(self.params_all)
                raise ValueError(errmes)
            return integ 
    
    
    def sigmalos_dequad(self,R_pc,n=1024,n_kernel=128,ignore_RuntimeWarning=True):
        return np.sqrt(self.sigmalos2_dequad(R_pc,n,n_kernel,ignore_RuntimeWarning))



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
        # return 1/np.log(len(self.data))
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
            # mes = f"lnposterior is nan. lnl:{lnl}, lnp_list:{lnp_list}"
            # mes += f"\np:{p}"
            # mes += f"\nargs:{args}"
            # mes += f"\nkwargs:{kwargs}"
            # mes += f"\nparams:{params}"
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
        # return (lnl + np.sum(lnp_list), lnl, *lnp_list)
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
        # if config is a file name, load it as a pandas.DataFrame.
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

        # self._sample = lambda size: np.random.uniform(self.data["lower"].values, self.data["upper"].values, size=size)

    def get_index(self,param_name):
        return self.data.index.get_loc(param_name)
    

    def extract_value_by_name(self,param,param_name):
        """get the value of the parameter from param (ndarray) based on the name of the parameter.

        Parameters
        ----------
        param: ndarray
            array of parameter values
        param_name: str
            name of the parameter

        Returns
        -------
        value: float
            value of the parameter
        """
        assert len(param) == len(self.data), f"len(param)={len(param)} != len(self.data)={len(self.data)}"
        return param[self.get_index(param_name)]


    def sample(self,size=None):
        # return self._sample(size=size)
        # return np.random.uniform(self.lower, self.upper, size=size)
        # if size is None:
        #     return np.random.uniform(self.lower, self.upper)
        # elif isinstance(size,int):
        #     size = (size,len(self.lower))
        #     return np.random.uniform(self.lower, self.upper, size=size)
        # elif isinstance(size,tuple):
        #     size = size + (len(self.lower),)
        #     return np.random.uniform(self.lower, self.upper, size=size)
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
    """
    required_param_names = []
    required_models = {}

    def __init__(self, dsph_name, show_init=False, submodels=None, **params):
        super().__init__(show_init, submodels or {}, **params)
        self._dsph_name = dsph_name
        self.load_config(dsph_name)

    @property
    def dsph_name(self):
        return self._dsph_name
    
    def load_config(self,dsph_name):
        if type(dsph_name) == str:
            if dsph_name == "Mock":
                self.logger.warning("dsph_name is 'Mock'. loc and scale must be set manually by 'reset_prior'.")
                loc,scale = np.nan, np.nan
            else:
                database = dsph_database.photometry.load_prior()
                loc,scale = database.set_index("name").T[dsph_name]
        else:
            # check if dsph_name is tuple of (loc,scale)
            self.logger.warning("dsph_name is not a string. Check if it is a tuple of (loc,scale).")
            loc,scale = dsph_name
            self.logger.info("loc: %r", loc)
            self.logger.info("scale: %r", scale)
        print_dict = {"log10_re_pc":loc,"e_log10_re_pc":scale}
        self.logger.info("%s:%r", self.__class__.__name__, print_dict)
        self.reset_prior(loc,scale)

    def reset_prior(self,loc,scale):
        self._lnprior = norm(loc=loc,scale=scale).logpdf
        self._sample = norm(loc=loc,scale=scale).rvs

    def sample(self,size):
        return self._sample(size=size)


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


    def load_data(self, dsph_type, dsph_name,shared=False):
        """ load dataset required for parameter fitting or estimation.
        data must be stored in self.data, as a pd.DataFrame.
        additional data must be stored in self.additional_data, as a dict.
        """
        self.dsph_name = dsph_name
        self.dsph_type = dsph_type
        self.shared = shared
        if self.dsph_type == "Mock":
            self.logger.warning("dsph_type is 'Mock'. 'data' attribute must be reset manually by 'reset_data'.")

        else:
            data = dsph_database.spectroscopy.load_kinematic_data(dsph_type,dsph_name)
            data = data.astype(self.dtype)  # decrease memory usage
            self.reset_data(data)


    def reset_data(self, data):
        """ reset data attribute and update FlatPriorModel.
        """
        self.data = data
        # override FlatPriorModel config by data
        # for vmem_kms
        self.logger.info("%s: Override FlatPriorModel config by data", self.__class__.__name__)
        self.logger.info("%s", "="*32)
        self.logger.info("%r", self["FlatPriorModel"].data.loc["vmem_kms"])
        lower = self.data["vlos_kms"].min()
        upper = self.data["vlos_kms"].max()
        # self["FlatPriorModel"].data.loc["vmem_kms"]["lower"] = lower
        # self["FlatPriorModel"].data.loc["vmem_kms"]["upper"] = upper
        self["FlatPriorModel"].data.loc["vmem_kms","lower"] = lower
        self["FlatPriorModel"].data.loc["vmem_kms","upper"] = upper
        self.logger.info("%s", "-"*8+">")
        self.logger.info("%r", self["FlatPriorModel"].data.loc["vmem_kms"])
        self.logger.info("%s", "="*32)


    @property
    def shared_memory_basename(self):
        if not self.shared:
            # raise ValueError("shared_memory_name is not available when shared is False.")
            return None
        return f"{self.dsph_type}_{self.dsph_name}"
    
    
    @property
    def data(self):
        if not self.shared:
            return self._data
        else:
            try:
                # If shared memory is already initialized, return the data.
                # print("debug: try to get shared memory.")
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
            
            

    # def get_shared_memory(self,suffix,size):
    #     if self.shared:
    #         name = self.shared_memory_basename+suffix
    #         try:
    #             # check if shared memory is already allocated.
    #             shm = SharedMemory(name=name,create=False)
    #             print(f"{self.__class__.__name__}: shared memory '{name}' is already allocated.")
    #             return shm
    #         except FileNotFoundError as e:
    #             # if shared memory is not allocated, allocate it.
    #             print(f"{self.__class__.__name__}: shared memory '{name}' is not allocated yet.")
    #             shm = SharedMemory(name=name,create=True, size=size)
    #             print(f"{self.__class__.__name__}: shared memory '{name}' is allocated.")
    #             return shm    
    #     else:
    #         raise ValueError("shared memory is not available when shared is False.")
    
    @property
    def n_data(self):
        return self._n_data
    
    @data.setter
    def data(self,data: pd.DataFrame):
        self._n_data = len(data)
        if not self.shared:
            # self._data = data
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
            
    
    # def __del__(self):
    #     if self.shared:
    #         self.release_shared_memory()
        

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
    


def get_default_estimation_model(dsph_type,dsph_name,
                                 config="priorconfig.csv",
                                 kwargs_load_data=None):
    """ return a default estimation model. 
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
        args_load_data=[dsph_type, dsph_name],
        kwargs_load_data=kwargs_load_data or {},
        submodels={
            "DSphModel" : DSphModel(submodels={
                "StellarModel" : PlummerModel(),
                "DMModel" : NFWModel(),
                "AnisotropyModel" : ConstantAnisotropyModel(),
            }),
            "FlatPriorModel": FlatPriorModel(config=config),
            "PhotometryPriorModel": PhotometryPriorModel(dsph_name)
        })
    return mdl



class KI17_Model:
    def __init__(self,params_KI17_Model):
        """
        params_KI17_model: pandas.Series, index = (params_DSphModel,params_FG_model,s)
        """
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    DMModel = NFWModel(a=2.78,b=7.78,g=0.675,rhos_Msunpc3=np.power(10,-2.05),rs_pc=np.power(10,3.96),R_trunc_pc=2000)
    myStellarModel = PlummerModel(re_pc=221)
    draco_model = DSphModel(submodels={"DMModel":DMModel,"StellarModel":myStellarModel},anib=1-np.power(10,0.13),dist_center_pc=76e3,ra_center_deg=0,de_center_deg=0)
    
    Rs = np.logspace(-1,9,100)
    ss = draco_model.sigmalos2(Rs)
    ss2 = [draco_model.sigmar2(R) for R in Rs]
    ss3 = [draco_model.naive_sigmalos2(R) for R in Rs]
    logger.info("%s", draco_model)
    logger.info("%r", draco_model.integrand_sigmalos2(1,1))
    plt.plot(Rs,np.sqrt(ss))
    plt.plot(Rs,np.sqrt(ss2))
    plt.plot(Rs,np.sqrt(ss3))
    plt.xscale("log")
    plt.yscale("log")
    #plt.ylim(0,40)
    plt.show()
    input("press any key")







