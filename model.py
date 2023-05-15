import pandas as pd
import numpy as np
import multiprocessing as multi
from copy import copy
import os
from numpy import array,pi,sqrt,exp,power,log,log10,log1p,cos,tan,sin, sort,argsort, inf, isnan
from scipy.stats import norm
from scipy.special import k0, betainc, beta, hyp2f1, erf, gamma, gammainc
from scipy import integrate
from scipy.constants import parsec, degree, physical_constants # parsec in meter, degree in radian
from scipy.integrate import quad
from scipy.interpolate import interp1d,Akima1DInterpolator

from multiprocessing import Pool
from abc import ABCMeta, abstractmethod
from functools import cached_property

import warnings

from .dequad import dequad
import dsph_database.spectroscopy
import dsph_database.photometry

GMsun_m3s2 = 1.32712440018e20
R_trunc_pc = 1866.

kg_eV = 1./physical_constants["electron volt-kilogram relationship"][0]
im_eV = 1./physical_constants["electron volt-inverse meter relationship"][0]
solar_mass_kg = 1.9884e30
C0 = (solar_mass_kg*kg_eV)**2*((1./parsec)*im_eV)**5
C1 = (1e9)**2 * (1e2*im_eV)**5
C_J = C0/C1


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
        __init__(show_init=False,submodels={},**params):
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
    

    def __init__(self,show_init=False,submodels={},**params):
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
        self.params = pd.Series({ p:np.nan for p in self.required_param_names})
        self.update("this",params)

        # # check if all required parameters are given.
        # if set(params.keys()) != set(self.required_param_names):
        #     raise ValueError(self.name+' has the paramsters: '+str(self.required_param_names)+" but input is "+str(params.keys()))
        
        # set model name
        if len(self.submodels) > 0:
            self.name += "_" + '+'.join((model.name for model in self.submodels.values))

        # check the consistency of params_all and required_param_names_combined
        params_all_index = [ pname.split(":")[-1] for pname in self.params_all.index]
        required_param_names_combined = self.required_param_names_combined 
        if not np.all(params_all_index == required_param_names_combined):
            raise ValueError("params_all and required_param_names_combined are inconsistent: "+str(params_all_index)+" vs "+str(required_param_names_combined))
            
        if show_init:
            print("initialized:")
            print(self.params_all)

    def __repr__(self):
        """ show model name and parameters.
        """
        ret = self.name + ":\n" + self.params_all.__repr__()
        #if len(self.submodels) > 0:
        #    ret += '\n'
        #    for model in self.submodels.values():
        #        ret += model.__repr__() + '\n'
        return ret
    

    @property
    def params_all(self):
        """ show all parameters as a pd.Series.
        """
        if len(self.submodels) == 0:
            return self.params
        else:
            # load submodels' parameters
            ret = [self.params]
            for model_name,model in self.submodels.items():
                _params = model.params_all.copy()
                # add submodels' name to the index of _params
                _params.index = [model_name+":"+index for index in _params.index]
                ret.append(_params)
            return pd.concat(ret)



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


    def update(self,target='all',new_params_dict=None,**kwargs):
        """ update model parameters recurrently. 
        If target is 'this', update parameters of this model.
        If target is 'all', update parameters of this model and submodels (sub-parameters).
        
        If there are unknown parameters in the new parameters, raise ValueError.

        Parameters
        ----------
        new_params_dict: dict or pd.Series, its key must be in self.required_param_names_combined.
            If it is None, new parameters are given by kwargs.
            If it is not None, kwargs are ignored.
        target: 'this' or 'all'.
            If 'this', update parameters of this model.
            If 'all', update parameters of this model and submodels (sub-parameters).
        **kwargs: new parameters' name and value. If both of new_params_dict and kwargs are given, raise ValueError.
        """
        # check if both of new_params_dict and kwargs are given
        if new_params_dict is not None and len(kwargs) > 0:
            raise ValueError("new_params_dict and kwargs cannot be given at the same time.")
        
        # convert new_params_dict to pd.Series if it is dict
        new_params = pd.Series(new_params_dict) if new_params_dict is not None else pd.Series(kwargs)
        
        # chech if new_params has unknown parameters
        if not np.all(np.isin(new_params.index,self.required_param_names_combined)):
            raise ValueError("new params has unknown parameters.\nunknown parameters:{}".format(new_params.index[~np.isin(new_params.index,self.required_param_names_combined)]))

        if target == 'this':
            # check if there are unknown parameters in the new parameters, not in self.required_param_names
            # if not np.all(self.is_required_param_names(new_params.index)):
            if not np.all(np.isin(new_params.index,self.required_param_names)):
                #raise ValueError("new params has unknown parameters.\nunknown parameters:{}".format(new_params.index[~self.is_required_param_names(new_params.index)]))
                raise ValueError("new params has unknown parameters.\nunknown parameters:{}".format(new_params.index[~np.isin(new_params.index,self.required_param_names)]))
            else:
                # update parameters of this model
                self.params.update(new_params)  # NOTE: self.params is a pd.Series
        elif target == 'all':
            # update parameters of this model and submodels
            # update parameters of this model
            new_params_in_this = new_params[new_params.index.isin(self.required_param_names)]
            if len(new_params_in_this) > 0:
                self.params.update(new_params_in_this)
            # update parameters of submodels
            for model_name,model in self.submodels.items():
                new_params_in_model = new_params[new_params.index.isin(model.required_param_names_combined)]
                if len(new_params_in_model) > 0:
                    model.update("all",new_params_in_model)
        else:
            raise ValueError("target must be 'this' or 'all'")

            

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
    def kernel(self,u,R,**kwargs):
        pass

    
class ConstantAnisotropyModel(AnisotropyModel):
    name = "ConstantAnisotropyModel"
    required_param_names = ['beta_ani']
    required_models = {}
    
    
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


    def sigmar2(self,r_pc):
        RELERROR_INTEG = 1e-6
        density_3d = self.submodels["StellarModel"].density_3d
        enclosure_mass = self.submodels["DMModel"].enclosure_mass
        f = self.submodels["AnisotropyModel"].f
        integrand = lambda r: density_3d(r)*f(r)*GMsun_m3s2*enclosure_mass(r)/r**2/f(r_pc)/density_3d(r_pc)*1e-6/parsec
        integ, abserr = integrate.quad(integrand,r_pc,np.inf)
        return integ

    
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
        
        density_3d = self.submodels["StellarModel"].density_3d
        density_2d = self.submodels["StellarModel"].density_2d
        enclosure_mass = self.submodels["DMModel"].enclosure_mass
        kernel = self.submodels["AnisotropyModel"].kernel
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

    def __init__(self,args_load_data=None,kwargs_load_data={},*args,**kwargs):
        """ initialize FittableModel. """
        super().__init__(*args,**kwargs)
        print("Fittable Model: args_load_data:")
        print(args_load_data)
        # check if args_load_data is a list.
        if not isinstance(args_load_data,list):
            raise TypeError('args_load_data must be a list.')
        print("Fittable Model: kwargs_load_data:")
        print(kwargs_load_data)
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
        return 1/np.log(len(self.data))
    

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
        self.update("all",params)
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
        self.update("all",params)
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
        self.update("all",params)
        lnp = self._lnpriors(*args,**kwargs)
        return lnp
        
    
    def lnposterior(self,p,*args,**kwargs):
        params = self.convert_params(p)
        self.update("all",params)
        lnl = -np.inf
        lnp_list = self._lnpriors(p,*args,**kwargs)
        if np.all([lnp > -np.inf for lnp in lnp_list]):
            lnl = self._lnlikelihood(*args,**kwargs)
        return (lnl + np.sum(lnp_list), lnl, *lnp_list)

    
    def lnposterior_wbic(self,p,*args,**kwargs):
        params = self.convert_params(p)
        self.update("all",params)
        lnl = -np.inf
        lnp_list = self._lnpriors(p,*args,**kwargs)
        if np.all([lnp > -np.inf for lnp in lnp_list]):
            lnl = self.lnlikelihood(*args,**kwargs) * self.inverse_temparature
        return (lnl + np.sum(lnp_list), lnl, *lnp_list)

    
    @cached_property
    def ndim(self):
        return len(self.params_all)



class FlatPriorModel(Model):
    """ flat prior model.
    """
    required_param_names = []
    required_models = {}

    def __init__(self, config, show_init=False, submodels={}, **params):
        super().__init__(show_init, submodels, **params)
        self.load_config(config)


    def load_config(self, config):
        """ load the upper and lower limits of each parameter from config.
        config: file name of the config file or pandas.DataFrame.
        NOTE: the upper and lower limits are for p (before conversion), not for params (after conversion).
        """
        # if config is a file name, load it as a pandas.DataFrame.
        if isinstance(config,str):
            self.data = pd.read_csv(config, index_col=0)
        else:
            self.data = config
            
        self.lower = self.data["lower"].values
        self.upper = self.data["upper"].values

        # self._sample = lambda size: np.random.uniform(self.data["lower"].values, self.data["upper"].values, size=size)


    def sample(self,size=None):
        # return self._sample(size=size)
        return np.random.uniform(self.lower, self.upper, size=size)


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
            return 0
        else:
            return -np.inf

        
        

class PhotometryPriorModel(Model):
    """ prior model for photometry.
    """
    required_param_names = []
    required_models = {}

    def __init__(self, dsph_name, show_init=False, submodels={}, **params):
        super().__init__(show_init, submodels, **params)
        self._dsph_name = dsph_name
        self.load_config(dsph_name)

    @property
    def dsph_name(self):
        return self._dsph_name
    
    def load_config(self,dsph_name):
        if type(dsph_name) == str:
            if dsph_name == "Mock":
                print("WARNING: dsph_name is 'Mock'. loc and scale must be set manually by 'reset_prior'.")
                loc,scale = np.nan, np.nan
            else:
                database = dsph_database.photometry.load_prior()
                loc,scale = database.set_index("name").T[dsph_name]
        else:
            # check if dsph_name is tuple of (loc,scale)
            print("WARNING: dsph_name is not a string. Check if it is a tuple of (loc,scale).")
            loc,scale = dsph_name
            print("loc:",loc)
            print("scale:",scale)
        print(f"{self.__class__.__name__}:log10_re_pc:{loc}\te_log10_re_pc:{scale}")
        self.reset_prior(loc,scale)

    def reset_prior(self,loc,scale):
        self._lnprior = norm(loc=loc,scale=scale).logpdf
        self._sample = norm(loc=loc,scale=scale).rvs

    def sample(self,size):
        return self._sample(size=size)



class SimpleDSphEstimationModel(FittableModel,Model):
    """ A Simple model for dwarf spheroidal galaxy, considering only kinematical dataset.
    """

    required_param_names = []
    required_models = {
        "DSphModel": DSphModel,
        "FlatPriorModel": FlatPriorModel,
        "PhotometryPriorModel": PhotometryPriorModel,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        print("Please check the consistensy of model parameters and config file.")
        comparison = {
            "config": self.submodels["FlatPriorModel"].data.index.tolist(),
            "params": self.required_param_names_combined,
        }
        try:
            print(pd.DataFrame(comparison))
            # check if comparison["config"] is consistent with comparison["params"] by backward matching.
            consistencies = [ (param in p) for p,param in zip(comparison["config"],comparison["params"])]
            assert all(consistencies)  # NOTE: We can find substring in string by using "in" operator.
        except ValueError as e:
            print(comparison)
            raise(e)
        except AssertionError as e:
            print(comparison)
            print(consistencies)
            raise(e)
        
        

    def convert_params(self, p):
        """ convert parameters from p to params. 
        Here, required_param_names_combined of this model is
            []
        """
        p_names = self.submodels["FlatPriorModel"].data.index.tolist()
        param_names = self.required_param_names_combined
        def convert_param(name,p):
            # check if "log10_" is in name by using a method of string
            if "log10_" in name:
                return 10**p
            elif "bfunc_" in name:
                # Inverse function of b -> log10(1-b)
                return 1 - 10**p
            else:
                return p
        d = { param_name:convert_param(p_name,p) for p_name,param_name,p in zip(p_names,param_names,p)}
        return pd.Series(d)


    def load_data(self, dsph_type, dsph_name):
        """ load dataset required for parameter fitting or estimation.
        data must be stored in self.data, as a pd.DataFrame.
        additional data must be stored in self.additional_data, as a dict.
        """
        self.dsph_name = dsph_name
        self.dsph_type = dsph_type
        if self.dsph_type == "Mock":
            print("WARNING: dsph_type is 'Mock'. 'data' attribute must be reset manually by 'reset_data'.")

        else:
            data = dsph_database.spectroscopy.load_kinematic_data(dsph_type,dsph_name)
            self.reset_data(data)

    def reset_data(self, data):
        """ reset data attribute and update FlatPriorModel.
        """
        self.data = data
        # override FlatPriorModel config by data
        # for vmem_kms
        print(self.__class__.__name__+":override FlatPriorModel config by data")
        print(self.__class__.__name__+":before:")
        print(self.submodels["FlatPriorModel"].data.loc["vmem_kms"])
        lower = self.data["vlos_kms"].values.min()
        upper = self.data["vlos_kms"].values.max()
        self.submodels["FlatPriorModel"].data.loc["vmem_kms"]["lower"] = lower
        self.submodels["FlatPriorModel"].data.loc["vmem_kms"]["upper"] = upper
        print(self.__class__.__name__+":after:")
        print(self.submodels["FlatPriorModel"].data.loc["vmem_kms"])

        

    def _lnlikelihoods(self):
        """ define natural logarithm of the likelihood function. """
        s2 = self.submodels["DSphModel"].sigmalos2_dequad(self.data.R_pc)
        err2 = self.data.e_vlos_kms**2
        vmem_kms = self.submodels["DSphModel"].params.vmem_kms
        return norm.logpdf(self.data.vlos_kms,loc=vmem_kms,scale=np.sqrt(s2+err2))
        

    def _lnpriors(self,p_before_conversion):
        """ define natural logarithm of the prior function. """
        idx_log10_re_pc = self.submodels["FlatPriorModel"].data.index.tolist().index("log10_re_pc")
        log10_re_pc = p_before_conversion[idx_log10_re_pc]
        return [
            self.submodels["FlatPriorModel"]._lnprior(p_before_conversion),
            self.submodels["PhotometryPriorModel"]._lnprior(log10_re_pc)
            ]


    prior_names = ["flat_prior","photometry_prior"]


    def sample(self,size=None):
        """ sample from the model. """
        p = self.submodels["FlatPriorModel"].sample(size)
        # override log10_re_pc
        idx_log10_re_pc = self.submodels["FlatPriorModel"].data.index.tolist().index("log10_re_pc")
        p[idx_log10_re_pc] = self.submodels["PhotometryPriorModel"].sample(size)
        return p
    


def get_default_estimation_model(dsph_type,dsph_name,config="priorconfig.csv"):
    mdl = SimpleDSphEstimationModel(
        args_load_data=[dsph_type, dsph_name],
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
    print(draco_model)
    print(draco_model.integrand_sigmalos2(1,1))
    plt.plot(Rs,np.sqrt(ss))
    plt.plot(Rs,np.sqrt(ss2))
    plt.plot(Rs,np.sqrt(ss3))
    plt.xscale("log")
    plt.yscale("log")
    #plt.ylim(0,40)
    plt.show()
    input("press any key")







