import pandas as pd
import numpy as np
import multiprocessing as multi

from .dequad import dequad

import os
from numpy import array,pi,sqrt,exp,power,log,log10,log1p,cos,tan,sin, sort,argsort, inf, isnan
from scipy.stats import norm
from scipy.special import k0, betainc, beta, hyp2f1, erf, gamma, gammainc
from scipy import integrate
from scipy.constants import parsec, degree # parsec in meter, degree in radian
from scipy.integrate import quad
from scipy.interpolate import interp1d,Akima1DInterpolator
from importlib.resources import files

from multiprocessing import Pool


DATA_DIR = files("jeanspy").joinpath("data")

GMsun_m3s2 = 1.32712440018e20
R_trunc_pc = 1866.



class Model:
    #params, required_params_name = pd.Series(), ['',]
    '''
    params, required_prams_name is undefined; must be defined in child class
    '''
    def __init__(self,show_init=False,submodels_dict={},**params):
        self.params = pd.Series(params)
        self.required_params_name = list(self.required_params_name)
        self.submodels = submodels_dict
        self.required_models = {}
        if 'submodels' in self.params.index:
            raise TypeError('submodels -> submodels_dict?')
        if set(self.params.index) != set(self.required_params_name):
            raise TypeError(self.name+' has the paramsters: '+str(self.required_params_name)+" but in put is "+str(self.params.index))
        for required_model in self.required_models:
            isinstance_ =  [isinstance(model,required_model) for model in self.submodels.values]
            if not any(isinstance_):
                raise TypeError("no "+str(required_model))
            
        if self.submodels != {}:
            self.name = ': ' + ' and '.join((model.name for model in self.submodels.values()))
        if show_init:
            print("initialized:")
            print(self.params_all)

    def __repr__(self):
        ret = self.name + ":\n" + self.params_all.__repr__()
        #if len(self.submodels) > 0:
        #    ret += '\n'
        #    for model in self.submodels.values():
        #        ret += model.__repr__() + '\n'
        return ret
    
    @property
    def params_all(self):
        ret = self.params
        if self.submodels != {}:
            ret = pd.concat([model.params_all for model in self.submodels.values()])
        return ret

    def show_required_params_name(self,models='all'):
        ret = self.required_params_name[:] # need copy because we must keep self.required_params_name
        if self.submodels != {} and models=='all':
            [ ret.extend(model.show_required_params_name('all')) for model in self.submodels.values() ]
        return ret

    def is_required_params_name(self,params_name_candidates):
        return [ (p in self.required_params_name) for p in params_name_candidates ]

    def update(self,new_params_dict=None,target='all',**kwargs):
        new_params = pd.Series(new_params_dict) if type(new_params_dict)==dict else None
        if new_params is None:
            new_params = pd.Series(kwargs)
        #print(np.isin(new_params.index, self.show_required_params_name('all')+['this','all']))
        if not np.any(np.isin(new_params.index, self.show_required_params_name('all')+['this','all'])):
            raise TypeError("new params has no required parameters.\nrequired parameters:{}".format(self.required_params_name))
        if target in ('this','all'):
            self.params.update(new_params)
            [model.params.update(new_params) for model in self.submodels.values()] if target in ('all',) else None
        else:
            [(model.params.update(new_params) if target==model.name else None) for model in self.submodels.values()]

            

class StellarModel(Model):
    name = "stellar Model"
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

    

class PlummerModel(StellarModel):
    name = "Plummer Model"
    required_params_name = ['re_pc',]
    
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
    required_params_name = ['re_pc','n']
    
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
    name = "Exp2dModel"
    required_params_name = ['re_pc',]
    
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
    name = "Exp3dModel"
    required_params_name = ['re_pc',]
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
    required_params_name = ['Rmax_pc',]
    def density_2d(self,R_pc):
        return 1./(pi*self.params.Rmax_pc**2)*np.ones_like(R_pc)
    def cdf_R(self,R_pc):
        return (R_pc/self.params.Rmax_pc)**2    

    
    
class DMModel(Model):
    name = "DM Model"

    
    
class NFWModel(DMModel):
    name = "NFW Model"
    required_params_name = ['rs_pc','rhos_Msunpc3','a','b','g','R_trunc_pc']
    
    #def __init__(self,params):
    #    super().__init__(params)
    #    if set(self.params.index) != set(NFW_params_name):
    #        raise TypeError('NFWModel has the paramsters: '+str(NFW_params_name))

    def mass_density_3d(self,r_pc):
        rs_pc, rhos_Msunpc3,a,b,g = self.params.rs_pc, self.params.rhos_Msunpc3, self.params.a, self.params.b,self.params.g
        x = r_pc/rs_pc
        return rhos_Msunpc3*power(x,-g)*power(1+power(x,a),-(b-g)/a)
        
    def enclosure_mass(self,r_pc):
        #ret = (array(r_pc.shape) if len(r_pc)>1 else 0)
        rs_pc, rhos_Msunpc3,a,b,g = self.params.rs_pc, self.params.rhos_Msunpc3, self.params.a, self.params.b,self.params.g
        
        x = power(r_pc/rs_pc,a)
        
        argbeta0 = (3-g)/a
        argbeta1 = (b-3)/a
        
        return (4.*pi*rs_pc**3*rhos_Msunpc3/a)*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,x/(1+x)) 
    
    def enclosure_mass_truncated(self,r_pc):
        #ret = (array(r_pc.shape) if len(r_pc)>1 else 0)
        rs_pc, rhos_Msunpc3,a,b,g = self.params.rs_pc, self.params.rhos_Msunpc3, self.params.a, self.params.b,self.params.g
        
        is_in_Rtrunc = r_pc<self.params.R_trunc_pc
        is_outof_Rtrunc = np.logical_not(is_in_Rtrunc)
        
        x = power(r_pc/rs_pc,a)*is_in_Rtrunc
        x_truncd = power(self.params.R_trunc_pc/rs_pc,a)
        argbeta0 = (3-g)/a
        argbeta1 = (b-3)/a
        
        #ret[is_in_Rtrunc] = (4.*pi*rs_pc**3*rhos_Msunpc3/a)*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,x/(1+x))
        #ret[is_outof_Rtrunc] = (4.*pi*rs_pc**3*rhos_Msunpc3/a)*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,x_truncd/(1+x_truncd))
        return is_in_Rtrunc * (4.*pi*rs_pc**3*rhos_Msunpc3/a)*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,x/(1+x)) + is_outof_Rtrunc * (4.*pi*rs_pc**3*rhos_Msunpc3/a)*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,x_truncd/(1+x_truncd))
        
        

class TruncatedNFWModel(DMModel):
    name = "NFW Model"
    required_params_name = ['rs_pc','rhos_Msunpc3','a','b','g','r_t_pc']
    
    #def __init__(self,params):
    #    super().__init__(params)
    #    if set(self.params.index) != set(NFW_params_name):
    #        raise TypeError('NFWModel has the paramsters: '+str(NFW_params_name))

    def mass_density_3d(self,r_pc):
        rs_pc, rhos_Msunpc3,a,b,g = self.params.rs_pc, self.params.rhos_Msunpc3, self.params.a, self.params.b,self.params.g
        x = r_pc/rs_pc
        return rhos_Msunpc3*power(x,-g)*power(1+power(x,a),-(b-g)/a)
        
    def enclosure_mass(self,r_pc):
        #ret = (array(r_pc.shape) if len(r_pc)>1 else 0)
        rs_pc, rhos_Msunpc3,a,b,g = self.params.rs_pc, self.params.rhos_Msunpc3, self.params.a, self.params.b,self.params.g
        r_t_pc = self.params.r_t_pc
        
        r_pc_trunc = r_pc.copy()
        r_pc_trunc[r_pc > r_t_pc] = r_t_pc
        
        x = power(r_pc_trunc/rs_pc,a)
        argbeta0 = (3-g)/a
        argbeta1 = (b-3)/a
        
        #ret[is_in_Rtrunc] = (4.*pi*rs_pc**3*rhos_Msunpc3/a)*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,x/(1+x))
        #ret[is_outof_Rtrunc] = (4.*pi*rs_pc**3*rhos_Msunpc3/a)*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,x_truncd/(1+x_truncd))
        return (4.*pi*rs_pc**3 * rhos_Msunpc3/a) * beta(argbeta0,argbeta1) * betainc(argbeta0,argbeta1,x/(1+x))
        
        

class DSphModel(Model):
    name = 'DSphModel'
    required_params_name = ['vmem_kms','anib']
    required_models = [StellarModel,DMModel]
    ncpu = multi.cpu_count()
#    def __init__(self,StellarModel,DMModel,**params_DSphModel):
#        """
#        params_DSphModel: pandas.Series, index = (params_StellarModel,params_DMModel,center_of_dSph)
#        """
#        # NOTE IT IS NOT COM{PATIBLE TO THE CODE BELOW!!!
#        super().__init__(**params_DSphModel)
#        self.submodels = (StellarModel,DMModel)
#        self.name = ' and '.join((model.name for model in self.submodels))
    def kernel_over_u(self,u,out="linear"):
        '''
        My kernel over u. Using 2F1.Using this kernel over u K(u)/u, sigmalos2 is given by
            sigmalos2(R) = 2 * \int_1^oo du \Sigma_\ast(uR)/\nu_\ast(R) * GM(uR) * K(u)/u.
            
        Descriptions:
            args: u = r_integrated/R, 1<u<oo
        '''
        anib = self.params.anib
        u2 = u**2
        if out=="linear":
            return 1/u * sqrt(1-1/u2)*((1.5-anib)*u2*hyp2f1(1.0,1.5-anib,1.5,1-u2)-0.5)
        elif out=="log":
            return -log(u) + log(1-1/u2)/2 + log((1.5-anib)*u2*hyp2f1(1.0,1.5-anib,1.5,1-u2)-0.5)

        
    def sigmar2(self,r_pc):
        RELERROR_INTEG = 1e-6
        anib = self.params.anib
        integrand = lambda r,r1: self.submodels["stellar_model"].density_3d(r)*np.power(r/r1,-2*anib)*GMsun_m3s2*self.submodels["dm_model"].enclosure_mass(r)/r**2/self.submodels["stellar_model"].density_3d(r_pc)*1e-6/parsec
        integ, abserr = integrate.quad(integrand,r_pc,np.inf,args=(r_pc,))
        return integ
    
    
    def sigmalos2_naive(self,R_pc):
        RELERROR_INTEG = 1e-6
        anib = self.params.anib
        stellar_model = self.submodels["stellar_model"]
        integrand = lambda r: (1-anib*(R_pc/r)**2)*stellar_model.density_3d(r)*self.sigmar2(r)/np.sqrt(1-(R_pc/r)**2)
        rs_interp = np.logspace(-2,6,51)
        integrand_interp = interp1d(rs_interp,[integrand(r) for r in rs_interp],kind="quadratic") 
        integ, abserr = integrate.quad(integrand_interp,R_pc,np.inf)
        return 2*integ/stellar_model.density_2d(R_pc)
    
    
    def integrand_sigmalos2(self,u,arg_R_pc):
        '''
        integrand of sigmalos2 at R = R_pc.
        u is a variable of integration, u=r/R.
        Domain: 1 < u < oo.
        '''
        stellar_model,dm_model = self.submodels["stellar_model"],self.submodels["dm_model"]
        nu = stellar_model.density_3d
        Sigma = stellar_model.density_2d
        M = dm_model.enclosure_mass
        u2,r = u**2,arg_R_pc*u
        # Note that parsec = parsec/m.
        # If you convert m -> pc,      ... var[m] * [1 pc/ parsec m] = var/parsec[pc].
        #                pc^1 -> m^pc, ... var[pc^1] * parsec(=[pc/m]) = var[m^-1]
        # Here var[m^3 pc^-1 s^-2] /parsec[m/pc] * 1e-6[km^2/m^2] = var[km^2/s^2]
        return 2 * self.kernel_over_u(u) * nu(r)/Sigma(arg_R_pc)*GMsun_m3s2 * M(r) / parsec * 1e-6

    
    def _sigmalos2(self,R_pc,n=2048):
        """
        calculate sigmalos2 for R_pc.
        R_pc: array, shape = (n,)
        """
        
        def func(u):
            u_ = np.array(u)[np.newaxis,:]  # axis: 1
            R_pc_ = np.array(R_pc)[:,np.newaxis]  # axis: 0
            return self.integrand_sigmalos2(u_,R_pc_)
        
        ret = dequad(func,1,inf,n=n,axis=1,
                     replace_nan_to_zero=True,
                     replace_inf_to_zero=True)
        
        if np.any(isnan(ret)):
            raise TypeError("nan! {}".format(ret))
            
        return ret
    
    
    def _sigmalos(self,R_pc,n=2048):
        return sqrt(self._sigmalos2(R_pc,n=n))
    
    
    
    def generate_R_interp(self,R_pc,n_interp=128,R_interp_extention=1.2):
        if n_interp >= len(R_pc):
            return R_pc
        
        log10_Rmin = log10(np.min(R_pc)/R_interp_extention)
        log10_Rmax = log10(np.max(R_pc)*R_interp_extention)
        R_pc_interp = np.logspace(log10_Rmin,log10_Rmax,n_interp)
        #R_pc_interp = np.concatenate([
        #    R_pc_interp,
        #    #np.linspace(1e-8,np.max(R_pc),n_interp),
        #    #np.random.choice(R_pc,min(n_interp,len(R_pc)),replace=False)+1e-8
        #])
        return R_pc_interp
    
    
    def sigmalos2(self,R_pc,n=1024,
                  R_pc_interp=None,
                  n_interp=64,R_interp_extention=1.2,kind="cubic"):
        if R_pc_interp is None:
            R_pc_interp = self.generate_R_interp(R_pc,n_interp,R_interp_extention)
        
        log_sigmalos2_func = interp1d(R_pc_interp,log(self._sigmalos2(R_pc_interp)),
                                      kind=kind,assume_sorted=True)
        return exp(log_sigmalos2_func(R_pc))
    
    
    def sigmalos(self,R_pc,n=1024,
                 R_pc_interp=None,
                 n_interp=64,R_interp_extention=1.2,kind="cubic"):
        if R_pc_interp is None:
            R_pc_interp = self.generate_R_interp(R_pc,n_interp,R_interp_extention)

        log_sigmalos_func = interp1d(R_pc_interp,log(self._sigmalos(R_pc_interp)),kind=kind)
        return exp(log_sigmalos_func(R_pc))

        
    

class KI17_Model:
    def __init__(self,params_KI17_Model):
        """
        params_KI17_model: pandas.Series, index = (params_DSphModel,params_FG_model,s)
        """
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dm_model = NFWModel(a=2.78,b=7.78,g=0.675,rhos_Msunpc3=np.power(10,-2.05),rs_pc=np.power(10,3.96),R_trunc_pc=2000)
    mystellar_model = plummer_Model(re_pc=221)
    draco_model = DSphModel(submodels_dict={"dm_model":dm_model,"stellar_model":mystellar_model},anib=1-np.power(10,0.13),dist_center_pc=76e3,ra_center_deg=0,de_center_deg=0)
    
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







