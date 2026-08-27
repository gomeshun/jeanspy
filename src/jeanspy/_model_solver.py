"""Classical spherical Jeans solver."""

from __future__ import annotations

from multiprocessing import cpu_count
import warnings

import numpy as np
from scipy import integrate
from scipy.constants import parsec

from ._model_anisotropy import AnisotropyModel
from ._model_base import Model
from ._model_jfactor import DMModel, GMsun_m3s2
from ._model_profiles import StellarModel
from .dequad import dequad


class DSphModel(Model):
    name = 'DSphModel'
    required_param_names = ["vmem_kms"]
    required_models = {
        "StellarModel": StellarModel,
        "DMModel": DMModel,
        "AnisotropyModel": AnisotropyModel
    }
    ncpu = cpu_count()


    def _sigmar2(self,r_pc):
        density_3d = self["StellarModel"].density_3d
        enclosed_mass = self["DMModel"].enclosed_mass
        f = self["AnisotropyModel"].f
        integrand = lambda r: density_3d(r)*f(r)*GMsun_m3s2*enclosed_mass(r)/r**2/f(r_pc)/density_3d(r_pc)*1e-6/parsec
        integ, _ = integrate.quad(integrand,r_pc,np.inf)
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
        enclosed_mass = self["DMModel"].enclosed_mass
        kernel = self["AnisotropyModel"].kernel
        r = R_pc*u
        # Note that parsec = parsec/m.
        # If you convert m -> pc,      ... var[m] * [1 pc/ parsec m] = var/parsec[pc].
        #                pc^1 -> m^pc, ... var[pc^1] * parsec(=[pc/m]) = var[m^-1]
        # Here var[m^3 pc^-1 s^-2] /parsec[m/pc] * 1e-6[km^2/m^2] = var[km^2/s^2]
        return 2.0 * kernel(u,R_pc,n=n_kernel)/u *  density_3d(r)/density_2d(R_pc)*GMsun_m3s2 * enclosed_mass(r) / parsec * 1e-6


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

    def sigmalos2(self,R_pc,n=1024,n_kernel=128,ignore_RuntimeWarning=True):
        """Return the line-of-sight velocity-dispersion squared.

        This is the backend-neutral entry point; the classical implementation
        uses its fixed-grid double-exponential (DE) ``dequad`` solver.
        """
        return self.sigmalos2_dequad(R_pc,n,n_kernel,ignore_RuntimeWarning)
