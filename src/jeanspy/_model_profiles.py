"""Stellar density profiles for the classical model backend."""

from __future__ import annotations

from abc import abstractmethod
from importlib.resources import files

import numpy as np
import pandas as pd
from numpy import exp, log, log10, pi, power
from scipy.interpolate import interp1d
from scipy.special import gamma, gammainc, k0

from ._model_base import Model


DATA_DIR = files("jeanspy").joinpath("data")


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
        r"""
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
        r'''
        cdf_R(R) = \int_0^R \dd{R'} 2\pi R' \Sigma(R')
        '''
        re_pc= self.params.re_pc
        return 1/(1+(re_pc/R_pc)**2)

    def mean_density_2d(self,R_pc):
        r'''
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
        df = pd.read_csv(DATA_DIR.joinpath("sersic_log10n_log10bn.csv"))
        self._b_interp = interp1d(df["log10n"].values,df["log10bn"].values,"cubic",assume_sorted=True)
        self.coeff = pd.read_csv(DATA_DIR.joinpath("coeff_dens.csv"), comment="#", delim_whitespace=True, header=None).values

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
        n = self.params.n
        return exp(-self.b*power(R_pc/self.params.re_pc,1/n))/self.norm

    def density_2d_normalized_re(self,R_pc):
        n = self.params.n
        return exp(-self.b*(power(R_pc/self.params.re_pc,1/n)-1))


    def cdf_R(self,R_pc):
        r'''
        cdf_R(R) = \int_0^R \dd{R'} 2\pi R' \Sigma(R')
        '''
        re_pc= self.params.re_pc
        n = self.params.n
        return gammainc(2*n,self.b*power(R_pc/re_pc,1/n)) # - gammainc(2*n,0)

    def mean_density_2d(self,R_pc):
        r'''
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
        r'''
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
        r'''
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
