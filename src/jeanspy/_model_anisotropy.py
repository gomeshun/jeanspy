"""Anisotropy profiles and line-of-sight kernels."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from numpy import power, sqrt
from scipy.special import hyp2f1

from ._model_base import Model
from .dequad import dequad


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
        r"""
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
        r"""
        u, R: 1d array
        """
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
        r"""
        kernel function K(u). LOSVD is given by

            sigmalos2(R) = 2 * \int_1^\infty du \nu_\ast(uR)/\Sigma_\ast(R) * GM(uR) * K(u)/u.

        # u: ndarray, shape = (n_u,)
        # R: ndarray, shape = (n_R,)

        return: ndarray, shape = (n_R,n_u)
        """
        n = 128 if ("n" not in kwargs) else kwargs["n"]
        u = np.asarray(u).reshape(-1)
        R = np.asarray(R).reshape(-1)

        u_expanded = u[np.newaxis,:,np.newaxis]  # axis = 1
        R_expanded = R[:,np.newaxis,np.newaxis]  # axis = 0
        def integrand(_u):
            return self.integrand_kernel(_u,R_expanded)

        integration = dequad(integrand,1,u_expanded,n,axis=2,replace_inf_to_zero=True,replace_nan_to_zero=True)  # shape = (n_R, n_u)

        return integration * self.f(R_expanded[...,0]*u_expanded[...,0])/u_expanded[...,0]
