# version 1.0.1
# 
# update of version 1.0.1
# - Use SkyCoord.radial_velocity_err as the err of the input array.
import warnings
import sys
if sys.version_info.minor < 7:
    warnings.warn("python version is older than 3.7, so ordinary dict cannot keep the ordering.\nHereafter we use OrderedDict instead of dict.")
    from collections import OrderedDict as dict

from . import mcgenerator, coord
from .dsph_model import DSphModel, PlummerModel, Exp2dModel, NFWModel, Uniform2dModel
from numpy import array,power,sqrt,log,exp,sin, nan, inf
from scipy.special import logsumexp
from scipy.stats import norm
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


#update likelihood
DEBUG = False

#dSph_property = pd.read_csv("dSph_property.csv",index_col=0)
#prior_lim = []
#draco_prop = dSph_property.loc["Draco"]
#sculptor_prop = dSph_property.loc["Sculptor"]
#RA0 = draco_prop.RAdeg
#DE0 = draco_prop.DEdeg
#DIST = draco_prop.DIST
#err_DIST = draco_prop.err_DIST
#dSph_property

def is_positive(*args):
    return array(args)>0

def pow10(x):
    return power(10,x)

class modKI17:
    '''
    interface for the likelihood calculation.
    
    
    Avalable methods:
    
        - __init__(sc_obsdata,sc_center0,paramlims_fname,prior_norm_fname):
            initialize the interface.
            
        - lnpost(params):
            return the log-posterior value for a given array of parameters.

    
    NOTE:
        - The order of the parameters are based on paramlims ("paramlims_fname").
        - This is a interface, so we cannot ascess the internal function
          (internal functions are prefixed by "_".)
        - The internal arguments of this class are "dict",
          simpler object in python than pd.Series.
    '''
    
    ######## initialization ########
    def __init__(self,sc_obsdata,sc_center0,model,paramlims_fname,prior_norm_fname):
        """
        initialize the properties of a modKI17 instance.
        
        parameters:
            - sc_obsdata:
                SkyCoord of observed data
            - sc_center0: 
                SkyCoord of ad hoc center of dSph
            - paramlims_fname
                Filename of prior configuration. File format: 
                    $     ,prms_min,prms_max
                    $ (p0),(p0_min),(p0_max)
                    $ (p1),(p1_loc),(p1_max)
                    $ ...
            - prior_norm_fname
                Filename of prior configuration. File format: 
                    $   ,loc   ,scale
                    $ p0,p0_loc,p0_scale
                    $ p1,p1_loc,p1_scale
                    $ ...
        """
        self.init_prior(paramlims_fname,prior_norm_fname)
        
        # initialize the param_names. 
        # Hereafter param_names becomes the basis of the parameter ordering.
        self.param_names = self.prior_lim.index
        print("order of parameters:\n",self.param_names)
        print("number of foreground components: ", self.n_components)
        
        self.init_data(sc_obsdata)
        
        self.init_center0(sc_center0)
        
        self.R_RoI = np.max(self.separation_pc(0,0,sc_center0.distance)) # use Rs.max as the RoI
        
        self.beta = 1/log(len(self.sc_obsdata))
        
        #print(Rs.describe())
        #print("beta: {}".format(self.beta)) if beta != 1 else None
        
        self.init_model(model)
    
    
    def init_prior(self,paramlims_fname,prior_norm_fname):
        # set the support of parameter
        self.prior_lim = pd.read_csv(paramlims_fname,index_col=0)
        self.prior_lim_param_names = self.prior_lim.index
        
        # set prior_norm of parameter
        _df_prior_norm = pd.read_csv(prior_norm_fname,index_col=0)
        self.prior_norm = norm(loc=_df_prior_norm["loc"],scale=_df_prior_norm["scale"])
        self.prior_norm_param_names = _df_prior_norm.index
    
    
    def init_data(self,sc_obsdata):
        self.sc_obsdata = sc_obsdata
        if hasattr(self.sc_obsdata,"radial_velocity_err"):
            print("sc_obsdata has radial_velocity_err. ",
                  "Likelihood function is defined with velocity error:",
                  self.sc_obsdata.radial_velocity_err
                 )
        else:
            self.sc_obsdata.radial_velocity_err = array(0)
        #self.sc_center0 = sc_center0
    
    
    def init_model(self,model):
        if model == "Plummer":
            mem = PlummerModel(re_pc=nan)
        elif model == "exp2d":
            mem = Exp2dModel(re_pc=nan)
        else: 
            raise TypeError("Undefined stellar model!")
        dm = NFWModel(
            a = nan, b = nan, g = nan,
            rhos_Msunpc3 = nan, rs_pc = nan,
            R_trunc_pc = nan
        )
        self.dsph = DSphModel(anib=nan,submodels_dict={"stellar_model":mem,"dm_model":dm},show_init=True)
        self.fg = Uniform2dModel(Rmax_pc=self.R_RoI,show_init=True)
    
    
    def init_center0(self,center0):
        self.sc_center0 = center0
        try:
            getattr(self.sc_center0,"distance_err")
        except AttributeError as e: 
            print(e)
            raise e
        

    ######## utils ########
    @property
    def n_components(self):
        '''
        return the number of foreground components based on the parameter list for the prior.
        '''
        if "sfg1" in self.param_names:
            return 3
        elif "sfg0" in self.param_names:
            return 2
        else:
            return 1
    
    
    @staticmethod
    def __params_to_series(index,**kwargs):
        """
        return the series whose elements are "kwargs" ordered by "index".
        it is useful to avoid to mistake the order of the parameters
        """
        sr = pd.Series(index=index)
        for key in kwargs:
            sr[key] = kwargs[key]
            #display(sr)
        display("params_to_ser",sr) if DEBUG else None
        return sr
    
    
    @staticmethod
    def __params_to_array(index,**kwargs):
        """
        return the series whoes elements are "kwargs" ordered by "index".
        it is useful to avoid to mistake the order of the parameters
        """
        ret = modKI17.params_to_series(index=index,**kwargs).values
        display("params_to_arr",ret) if DEBUG else None
        return ret
    
    
    @staticmethod
    def __array_to_series(index,params):
        ret = pd.Series(params,index=index)
        display("array_to_ser",ret) if DEBUG else None
        return ret
    
    
    @staticmethod
    def array_to_dict(index,params):
        '''
        convert a parameter array to dict form, whose keys are based on "index".
        '''
        if len(index) != len(params):
            raise TypeError("mismatch of parameter length!\n\
                             \tindex:{}\n\tparams:{}".format(index,params))
        return {key:val for key,val in zip(index,params)}
    
    
    @staticmethod
    def dict_to_array(params):
        '''
        convert a parameter dict to numpy array form.
        '''
        return array(list(params.values()))
    
    
    ######## utils ########
    def sc_center(self,dra0,dde0,dist):
        return SkyCoord(
            ra=self.sc_center0.ra+dra0*u.deg,
            dec=self.sc_center0.dec+dde0*u.deg,
            distance=dist*u.pc)

    
    def separation_pc(self,dra0,dde0,dist):
        c = self.sc_center(dra0,dde0,dist)
        return c.distance.value*np.sin(self.sc_obsdata.separation(c).rad)
    
    
    def memfg_ratio_at_R(self,mem,odds,R):
        return 1/(1+ 1/(odds * mem.density_2d_normalized_re(R)))

    
    ######## statistics ########
    def is_parameters_in_domain(self,p):
        """
        check the parameters are in valid domain.
        
        params: dict
        """
        p_in = self.dict_to_array(p)
        p_lo, p_hi = self.prior_lim.prms_min.values, self.prior_lim.prms_max.values
        if DEBUG:
            print(p_lo,p_in,p_hi)
        is_in_minmax = np.all((p_lo < p_in) & (p_in < p_hi))
        #is_ordered = prms_df.inprms.vfg0 < prms_df.inprms.vfg1
        
        if self.n_components == 3:
            sfg0,sfg1 = p["sfg0"],p["sfg1"]
            sfg2 = 1-sfg0-sfg1
            is_ordered = (sfg0 > sfg1) & (sfg1 > sfg2) & (sfg2 > 0)
        elif self.n_components == 2:
            sfg0 = p["sfg0"]
            sfg1 = 1-sfg0
            is_ordered = (sfg0 > sfg1) & (sfg1 > 0)
        else: # self.n_components == 1:
            is_ordered = True
        if DEBUG:
            display("is_in_minmax",(p_lo < p_in) & (p_in < p_hi),"is_ordered",is_ordered)
        return is_in_minmax & is_ordered

    
    def _lnprior(self,p):
        '''
        p: dict of parameters
        '''
        if not self.is_parameters_in_domain(p):
            if DEBUG:
                display("out ouf dom:",self.is_parameters_in_domain(p))
            return -inf
        else:
            # extract prior parameters
            args_prior = array([p[p_name] for p_name in self.prior_norm_param_names]) 
            logGs = self.prior_norm.logpdf(args_prior)
            #logGs = []
            #logGs.append(norm.logpdf(re,  loc=191,      scale=5.7)     )
            #logGs.append(norm.logpdf(odds,loc=8.794,    scale=0.5107)  )
            #logGs.append(norm.logpdf(dra0,loc=4.212e-3, scale=7.052e-3))
            #logGs.append(norm.logpdf(dde0,loc=-1.991e-3,scale=3.302e-3))
            #logGs.append(norm.logpdf(dist,loc=self.sc_center0.distance,scale=self.sc_center0.distance_err))
            return np.sum(logGs)

        
#    def __call__(self,**args):
#        '''
#        return the log-posterior value for a given dictionarry of the parameters.
#        '''
#        params = self.params_to_array(index=self.param_names,**args)
#        ret = self.lnprior(params)
#        if ret == -inf:
#            return ret
#        else:
#            return ret + np.sum(self.lnlikeli(params)) 
    
    
    def _lnpost(self,p):
        '''
        p: dict
        '''
        ret = self._lnprior(p)
        if ret == -inf:
            return ret
        else:
            return ret + np.sum(self._lnlikeli(p)) 
        
        
    def lnpost(self,params):
        '''
        return the log-posterior value for a given parameter array.
        
        params: array of parameters
        '''
        p = self.array_to_dict(index=self.param_names,params=params)
        return self._lnpost(p)
    
    
    def lnposterior_general(self,p):
        lnp = self._lnprior(p)
        if lnp > -inf:
            lnl = self._lnlikeli(p) 
            return (self.beta*lnl+lnp, lnl)
        else:
            return (-inf, nan)
    

    def _lnfmems(self,p,vs=None,vobs_err=None,with_Rs=False,with_s_R=False):
        '''
        p: dict of the parameters
        
        Note for numpy broadcasting
            vs                res            # Note
            (n_star,)      -> (n_star,)      # sigmalos.shape == Rs.shape == (n_star,)
            (n_v, 1)       -> (n_v, n_star)  # (n_v, 1     ) * (n_star,) -> (n_v, n_star)
            (n_v, n_star)  -> (n_v, n_star)  # (n_v, n_star) * (n_star,) -> (n_v, n_star)
        '''

        if DEBUG:
            display("args of loglikeli:",p)
        
        vs = (self.sc_obsdata.radial_velocity.value if vs is None else vs)
        vobs_err = (self.sc_obsdata.radial_velocity_err.value if vobs_err is None else vobs_err) 
        mem,dm= self.dsph.submodels["stellar_model"],self.dsph.submodels["dm_model"]
        
        # update parameters
        # Note that re_pc given by the stelar fit is just the angle (re_rad), not re_pc !!!
        mem.update({"re_pc":p["re_pc"]*(p["dist"]/self.sc_center0.distance.value)}) 
        dm.update({"rs_pc":pow10(p["log10_rs_pc"]),"rhos_Msunpc3":pow10(p["log10_rhos_Msunpc3"]),
                   "a":p["a"],"b":p["b"],"g":p["g"]})
        self.dsph.update({"anib":1-pow10(-p["mlog10_1manib"])})
        ref_R = mem.half_light_radius() # 1.67834699001666*re
        
        Rs = self.separation_pc(p["dra0"],p["dde0"],p["dist"]) # here 
        
        sigmalos = self.dsph.sigmalos_dequad_interp1d_downsampled(Rs,)
        #sigmalos = self.dsph.sigmalos_dequad(Rs)
        
        ret = norm.logpdf(vs,loc=p["vmem"],scale=sqrt(sigmalos**2+vobs_err**2))
        
        if (not with_Rs) and (not with_s_R):
            return ret
        else:
            ret = {"lnfmems":ret}
            if with_Rs:
                ret["Rs"] = Rs
            if with_s_R:
                ret["s_R"] = 1/(1+ 1/(p["odds"] * mem.density_2d_normalized_re(Rs))) # Not s but s(R)
            return ret
    
    
    def _lnffgs(self,p,vs=None,vobs_err=None):
        vs = (self.sc_obsdata.radial_velocity.value if vs is None else vs)
        vobs_err = (self.sc_obsdata.radial_velocity_err.value if vobs_err is None else vobs_err) 
        
        lnffg = lambda i: norm.logpdf(vs,loc=p["vfg{}".format(i)],scale=sqrt(p["dvfg{}".format(i)]**2+vobs_err**2))
        lnffgs = [lnffg(i) for i in range(self.n_components)]
        
        return lnffgs
    
#    def _lnlikeli(self,params):
#        '''
#        return the log-likelihood value for a given parameter array.
#        params: dict
#        '''
#        return self._loglikelihood(**params)
    
    
    def _sfgs(self,p):
        '''
        return sfgs (sfg0, sfg1, sfg2).
        The number of the return is altered based on "n_components".
        '''
        _sfg_raw = [p["sfg{}".format(i)] for i in range(self.n_components - 1)]
        return array([*_sfg_raw, 1 - np.sum(_sfg_raw)])
    
    
    def _lnlikelis(self,p,vs=None,vobs_err=None): 
        '''
        return log-likelihood value.
        
        p: dict
        
        Note for numpy broadcasting
            vs                res            # Note
            (n_star,)      -> (n_star,)      # sigmalos.shape == Rs.shape == (n_star,)
            (n_v, 1)       -> !Error!        # logfmem.shape == (n_v, n_star), while logffgs.shape == (n_components, n_v, 1 )
            (n_v, n_star)  -> (n_v, n_star)  # (n_v, n_star) * (n_star,) -> (n_v, n_star)
        '''
        # R_trunc_pc is fixed 2000 pc but its not affect to the result (truncation radius is not used in the calculation of sigmalos in "dSph_Model")

        if DEBUG:
            display("args of loglikeli:",p)
        
        vs = (self.sc_obsdata.radial_velocity.value if vs is None else vs)
        vobs_err = (self.sc_obsdata.radial_velocity_err.value if vobs_err is None else vobs_err) 
        
        mem = self._lnfmems(p,vs,vobs_err,with_Rs=True,with_s_R=True)
        logfmem,Rs,s_R = mem["lnfmems"], mem["Rs"], mem["s_R"]
        logffgs = self._lnffgs(p,vs,vobs_err)

        logfs = np.array([logfmem,*logffgs])  # logfs.shape == (n_compontnetns+1, res.shape), res.shape[-1] == n_star.
        
        #s_R = 1/(1+ 1/(p["odds"] * mem.density_2d_normalized_re(Rs))) # Not s but s(R)
        ss = np.array([s_R, *((1-s_R)[np.newaxis,:]*self._sfgs(p)[:,np.newaxis])])  # ss.shape == (n_components+1, n_star)
        if logfs.shape != ss.shape:
            ss = ss[:,np.newaxis]  # if vs.shape == (n_v, n_star), ss.shape -> (n_components+1, 1, n_star)
        
        if DEBUG:
            display("sigmalos:{}".format(sigmalos))
            print("fmem:{}".format(fmem))
            print("s*fmem+(1-s)*ffg:{}".format(s_R*fmem+(1-s_R)*ffg))
        
        #print(np.array(logfs).shape,np.array(ss).shape)
        #for logf in logfs:
        #    print("DEBUG:\nlogf:{}".format(logf))
        #for s in ss:
        #    print("DEBUG:\ns:{}".format(s))
        
        ret = logsumexp(a=logfs,b=ss,axis=0) # note that logsumexp must be used to aviod over/underflow of the likelihood
        
        return ret


    def _lnlikeli(self,p):
        return np.sum(self._lnlikelis(p))

    
    def weighted_distribution_functions(self,p):
        '''
        p: dict of the parameters
        '''
        
        if DEBUG:
            display("args of loglikeli:",p)
        
        vs=self.sc_obsdata.radial_velocity.value        
        vobs_err = self.sc_obsdata.radial_velocity_err.value
        
        _logfmem = self._lnfmems(p,with_s_R=True) # here we update mem.
        logfmem,s_R = _logfmem["lnfmems"], _logfmem["s_R"]
        logffgs = self._lnffgs(p)
        logfs = [logfmem,*logffgs]
        
        mem = self.dsph.submodels["stellar_model"]
        
        sfgs = self._sfgs(p)
        #s_R = 1/(1+ 1/(p["odds"] * mem.density_2d_normalized_re(Rs))) # Not s but s(R)
        ss = [s_R, *((1-s_R)[np.newaxis,:]*self._sfgs(p)[:,np.newaxis])]
        
        if DEBUG:
            display("sigmalos:{}".format(sigmalos))
            print("fmem:{}".format(fmem))
            print("s*fmem+(1-s)*ffg:{}".format(s*fmem+(1-s)*ffg))
        
        ret = array(ss)*exp(logfs)
        
        return ret
    
    
    def membership_prob(self,p):
        params = pd.Series(locals()).drop("self")
        weighted_dist_funcs = self.weighted_distribution_functions(p)  # (n_distfunc, n_star)
        normalizations = weighted_dist_funcs.sum(axis=0)
        memfg_ratios = (weighted_dist_funcs/normalizations)[0]  # (n_distfunc, n_star)
        return memfg_ratios

    
    
class modKI17_1gauss(modKI17):
    def loglikelihood(self,re_pc,odds,dra0,dde0,
            log10_rs_pc,log10_rhos_Msunpc3,a,b,g,
            mlog10_1manib,
            vmem,vfg0,dvfg0,dist
    ):
        mem = self.dsph.submodels["stellar_model"]
        mem.update({"re_pc":re_pc*(dist/self.sc_center0.distance.value)}) # TODO: "update" is required. This is very confusing and may cause bugs.
        
        vs = self.sc_obsdata.radial_velocity.value
        vobs_err = (self.sc_obsdata.radial_velocity_err.value if hasattr(self.sc_obsdata,"radial_velocity_err") else 0)
        s = 1/(1+ 1/(odds * mem.density_2d_normalized_re(self.separation_pc(dra0,dde0,dist)))) # In this line, we use "mem.foo", so we should update "mem" in advance.
        
        lnfmems = self.lnfmems(re_pc,dra0,dde0,log10_rs_pc,log10_rhos_Msunpc3,a,b,g,mlog10_1manib,vmem,dist)
        lnffg0s = norm.logpdf(vs,loc=vfg0,scale=sqrt(dvfg0**2+vobs_err**2))
    
        logfs = [lnfmems,lnffg0s]
        ss = [s,(1-s)]
        
        ret = np.sum(logsumexp(a=logfs,b=ss,axis=0)) # note that logsumexp must be used to aviod over/underflow of the likelihood
        
        return ret

class modKI17_memonly:
    '''
    interface for the likelihood calculation.
    
    
    Avalable methods:
    
        - __init__(sc_obsdata,sc_center0,paramlims_fname,prior_norm_fname):
            initialize the interface.
            
        - lnpost(params):
            return the log-posterior value for a given array of parameters.

    
    NOTE:
        - The order of the parameters are based on paramlims ("paramlims_fname").
        - This is a interface, so we cannot ascess the internal function
          (internal functions are prefixed by "_".)
        - The internal arguments of this class are "dict",
          simpler object in python than pd.Series.
    '''
    
    ######## initialization ########
    def __init__(self,sc_obsdata,sc_center0,model,paramlims_fname,prior_norm_fname):
        """
        initialize the properties of a modKI17 instance.
        
        parameters:
            - sc_obsdata:
                SkyCoord of observed data
            - sc_center0: 
                SkyCoord of ad hoc center of dSph
            - paramlims_fname
                Filename of prior configuration. File format: 
                    $     ,prms_min,prms_max
                    $ (p0),(p0_min),(p0_max)
                    $ (p1),(p1_loc),(p1_max)
                    $ ...
            - prior_norm_fname
                Filename of prior configuration. File format: 
                    $   ,loc   ,scale
                    $ p0,p0_loc,p0_scale
                    $ p1,p1_loc,p1_scale
                    $ ...
        """
        self.init_prior(paramlims_fname,prior_norm_fname)
        
        # initialize the param_names. 
        # Hereafter param_names becomes the basis of the parameter ordering.
        self.param_names = self.prior_lim.index
        print("order of parameters:\n",self.param_names)
        
        self.init_data(sc_obsdata)
        self.init_center0(sc_center0)
        
        self.R_RoI = np.max(self.separation_pc(0,0,sc_center0.distance)) # use Rs.max as the RoI
        self.beta = 1/log(len(self.sc_obsdata))
        
        #print(Rs.describe())
        #print("beta: {}".format(self.beta)) if beta != 1 else None
        
        self.init_model(model)
    
    
    def init_prior(self,paramlims_fname,prior_norm_fname):
        # set the support of parameter
        self.prior_lim = pd.read_csv(paramlims_fname,index_col=0)
        self.prior_lim_param_names = self.prior_lim.index
        
        # set prior_norm of parameter
        _df_prior_norm = pd.read_csv(prior_norm_fname,index_col=0)
        self.prior_norm = norm(loc=_df_prior_norm["loc"],scale=_df_prior_norm["scale"])
        self.prior_norm_param_names = _df_prior_norm.index
    
    
    def init_data(self,sc_obsdata):
        self.sc_obsdata = sc_obsdata
        if hasattr(self.sc_obsdata,"radial_velocity_err"):
            print("sc_obsdata has radial_velocity_err. ",
                  "Likelihood function is defined with velocity error:",
                  self.sc_obsdata.radial_velocity_err
                 )
        else:
            self.sc_obsdata.radial_velocity_err = array(0) * u.km/u.second
        #self.sc_center0 = sc_center0
    
    
    def init_model(self,model):
        if model == "Plummer":
            mem = PlummerModel(re_pc=nan)
        elif model == "exp2d":
            mem = Exp2dModel(re_pc=nan)
        else: 
            raise TypeError("Undefined stellar model!")
        dm = NFWModel(
            a = nan, b = nan, g = nan,
            rhos_Msunpc3 = nan, rs_pc = nan,
            R_trunc_pc = nan
        )
        self.dsph = DSphModel(anib=nan,submodels_dict={"stellar_model":mem,"dm_model":dm},show_init=True)

    
    
    def init_center0(self,center0):
        self.sc_center0 = center0
        try:
            getattr(self.sc_center0,"distance_err")
        except AttributeError as e: 
            print(e)
            raise e
        

    ######## utils ########
    @staticmethod
    def __params_to_series(index,**kwargs):
        """
        return the series whose elements are "kwargs" ordered by "index".
        it is useful to avoid to mistake the order of the parameters
        """
        sr = pd.Series(index=index)
        for key in kwargs:
            sr[key] = kwargs[key]
            #display(sr)
        display("params_to_ser",sr) if DEBUG else None
        return sr
    
    
    @staticmethod
    def __params_to_array(index,**kwargs):
        """
        return the series whoes elements are "kwargs" ordered by "index".
        it is useful to avoid to mistake the order of the parameters
        """
        ret = modKI17.params_to_series(index=index,**kwargs).values
        display("params_to_arr",ret) if DEBUG else None
        return ret
    
    
    @staticmethod
    def __array_to_series(index,params):
        ret = pd.Series(params,index=index)
        display("array_to_ser",ret) if DEBUG else None
        return ret
    
    
    @staticmethod
    def array_to_dict(index,params):
        '''
        convert a parameter array to dict form, whose keys are based on "index".
        '''
        if len(index) != len(params):
            raise TypeError("mismatch of parameter length!\n\
                             \tindex:{}\n\tparams:{}".format(index,params))
        return {key:val for key,val in zip(index,params)}
    
    
    @staticmethod
    def dict_to_array(params):
        '''
        convert a parameter dict to numpy array form.
        '''
        return array(list(params.values()))
    
    
    ######## utils ########
    def sc_center(self,dra0,dde0,dist):
        return SkyCoord(
            ra=self.sc_center0.ra+dra0*u.deg,
            dec=self.sc_center0.dec+dde0*u.deg,
            distance=dist*u.pc)

    
    def separation_pc(self,dra0,dde0,dist):
        c = self.sc_center(dra0,dde0,dist)
        return c.distance.value*np.sin(self.sc_obsdata.separation(c).rad)
    
    
    ######## statistics ########
    def is_parameters_in_domain(self,p):
        """
        check the parameters are in valid domain.
        
        params: dict
        """
        p_in = self.dict_to_array(p)
        p_lo, p_hi = self.prior_lim.prms_min.values, self.prior_lim.prms_max.values
        if DEBUG:
            print(p_lo,p_in,p_hi)
        is_in_minmax = np.all((p_lo < p_in) & (p_in < p_hi))
        #is_ordered = prms_df.inprms.vfg0 < prms_df.inprms.vfg1
        
        if DEBUG:
            display("is_in_minmax",(p_lo < p_in) & (p_in < p_hi),"is_ordered",is_ordered)
        return is_in_minmax

    
    def _lnprior(self,p):
        '''
        p: dict of parameters
        '''
        if not self.is_parameters_in_domain(p):
            if DEBUG:
                display("out ouf dom:",self.is_parameters_in_domain(p))
            return -inf
        else:
            # extract prior parameters
            args_prior = array([p[p_name] for p_name in self.prior_norm_param_names]) 
            logGs = self.prior_norm.logpdf(args_prior)
            #logGs = []
            #logGs.append(norm.logpdf(re,  loc=191,      scale=5.7)     )
            #logGs.append(norm.logpdf(odds,loc=8.794,    scale=0.5107)  )
            #logGs.append(norm.logpdf(dra0,loc=4.212e-3, scale=7.052e-3))
            #logGs.append(norm.logpdf(dde0,loc=-1.991e-3,scale=3.302e-3))
            #logGs.append(norm.logpdf(dist,loc=self.sc_center0.distance,scale=self.sc_center0.distance_err))
            return np.sum(logGs)

        
#    def __call__(self,**args):
#        '''
#        return the log-posterior value for a given dictionarry of the parameters.
#        '''
#        params = self.params_to_array(index=self.param_names,**args)
#        ret = self.lnprior(params)
#        if ret == -inf:
#            return ret
#        else:
#            return ret + np.sum(self.lnlikeli(params)) 
    
    
    def _lnpost(self,p):
        '''
        p: dict
        '''
        ret = self._lnprior(p)
        if ret == -inf:
            return ret
        else:
            return ret + np.sum(self._lnlikeli(p)) 
        
        
    def lnpost(self,params):
        '''
        return the log-posterior value for a given parameter array.
        
        params: array of parameters
        '''
        p = self.array_to_dict(index=self.param_names,params=params)
        return self._lnpost(p)
    
    
    def lnposterior_general(self,p):
        lnp = self._lnprior(p)
        if lnp > -inf:
            lnl = self._lnlikeli(p) 
            return (self.beta*lnl+lnp, lnl)
        else:
            return (-inf, nan)
    

    def _lnfmems(self,p,vs=None,vobs_err=None,with_Rs=False,with_s_R=False):
        '''
        p: dict of the parameters
        '''

        if DEBUG:
            display("args of loglikeli:",p)
        
        vs = (self.sc_obsdata.radial_velocity.value if vs is None else vs)
        vobs_err = (self.sc_obsdata.radial_velocity_err.value if vobs_err is None else vs) 
        mem,dm= self.dsph.submodels["stellar_model"],self.dsph.submodels["dm_model"]
        
        # update parameters
        # Note that re_pc given by the stelar fit is just the angle (re_rad), not re_pc !!!
        mem.update({"re_pc":p["re_pc"]*(p["dist"]/self.sc_center0.distance.value)}) 
        dm.update({"rs_pc":pow10(p["log10_rs_pc"]),"rhos_Msunpc3":pow10(p["log10_rhos_Msunpc3"]),
                   "a":p["a"],"b":p["b"],"g":p["g"]})
        self.dsph.update({"anib":1-pow10(-p["mlog10_1manib"])})
        ref_R = mem.half_light_radius() # 1.67834699001666*re
        
        Rs = self.separation_pc(p["dra0"],p["dde0"],p["dist"]) # here 
        
        sigmalos = self.dsph.sigmalos_dequad_interp1d_downsampled(Rs,iteration=3)
        #sigmalos = self.dsph.sigmalos_dequad(Rs)
        
        ret = norm.logpdf(vs,loc=p["vmem"],scale=sqrt(sigmalos**2+vobs_err**2))
        
        if (not with_Rs) and (not with_s_R):
            return ret
        else:
            ret = {"lnfmems":ret}
            if with_Rs:
                ret["Rs"] = Rs
            if with_s_R:
                ret["s_R"] = 1/(1+ 1/(p["odds"] * mem.density_2d_normalized_re(Rs))) # Not s but s(R)
            return ret
    
    
    def _lnlikelis(self,p,vs=None,vobs_err=None): 
        '''
        return log-likelihood value.
        
        p: dict
        '''
        # R_trunc_pc is fixed 2000 pc but its not affect to the result (truncation radius is not used in the calculation of sigmalos in "dSph_Model")

        if DEBUG:
            display("args of loglikeli:",p)
        
        vs = (self.sc_obsdata.radial_velocity.value if vs is None else vs)
        vobs_err = (self.sc_obsdata.radial_velocity_err.value if vobs_err is None else vs) 
        
        logfmem = self._lnfmems(p,vs,vobs_err)
        
        
        ret = logfmem # note that logsumexp must be used to aviod over/underflow of the likelihood
        
        return ret


    def _lnlikeli(self,p):
        return np.sum(self._lnlikelis(p))



 
class modKI17_photometry:
    def __init__(self,data,center0,
                 #area_fname,area_config_fname,
                 #nrow=10000,
                 model="Plummer"):
        '''
        modKI17 likelihood for the photometric observation.
        
        Parameters:
            data       : SkyCoord of observed stars
            center0    : SkyCoord of default center position
            # area_fname : filename of random sampled area
            # area_config_fname: filename of random sampled area
            # n_row      : number of the row you want to use in "area_fname".
        '''
        self.obs_data = data
        self.center0 = center0
        self.R_RoI = self.center0.distance.pc * sin(center0.separation(data).rad.max())
        
        if model=="Plummer":
            self.mem = PlummerModel(re_pc=200)
        elif model=="exp2d":
            self.mem = Exp2dModel(re_pc=200)
        else: 
            raise TypeError("Undefined stellar model!")
        self.fg = Uniform2dModel(Rmax_pc=self.R_RoI)
        
        #df_sample = pd.read_csv(area_fname)[:nrow]
        #self.sc_sample = SkyCoord(ra=df_sample.ra*u.deg,dec=df_sample.dec*u.deg)
        #self.df_sample_config  = pd.read_csv(area_config_fname,squeeze=True,index_col=0,header=None)
        # For the general case, R_RoI is ambiguous
    
    @property
    def beta(self):
        return 1/log(len(self.obs_data))
        
    def _separation_pc(self,sc,dra0,dde0):
        center = SkyCoord(ra=self.center0.ra+dra0*u.deg,dec=self.center0.dec+dde0*u.deg,distance=self.center0.distance)
        return self.center0.distance.pc*np.sin(sc.separation(center).rad)
    
    def Rs(self,dra0,dde0):
        return self._separation_pc(sc=self.obs_data,dra0=dra0,dde0=dde0)
    
    def __call__(self,prms):
        if self.lnprior(*prms) == -inf:
            return -inf
        else:
            return np.sum(self.loglikelis(*prms))
        
    def lnpost(self,prms):
        if self.lnprior(*prms) == -inf:
            return -inf
        else:
            return self.loglikeli(*prms)
    
    def loglikeli(self,re,odds,dra0,dde0):
        #return np.sum(self.loglikelis(re,odds,dra0,dde0))
        logfmem,logffg,logftot,s = self.logfs(re,odds,dra0,dde0)
        return np.sum(logftot)
    
    def lnprior(self,re,odds,dra0,dde0):
        if re<0 or odds<0 or np.abs(dra0)>1 or np.abs(dde0)>1:
            return -inf
        else:
            return 0
    
    def logfs(self,re,odds,dra0,dde0,Rs=None):
        '''
        Return f_mem and f_fg
        '''
        mem,fg = self.mem,self.fg
        mem.update({"re_pc":re})
        ref_R = self.mem.half_light_radius() # 1.67834699001666*re for sersic model
        
        # C1 and C0 are the normalization factor of the member and foreground distribution.
        # For general area case, which is given by
        # \int_Omega d\Omega ~ Omega * (mean of p(R_i), where R_i denotes each samplinig points) 
        #C0 = mem.density_2d_trunc()
        #logC0 = log(C0)
        #C1_0 = 1 / np.mean(mem.density_2d(Rs_rand)) # C1/C0
        #logC1_0 = -(logsumexp(mem.logdensity_2d(Rs_rand)) - log(len(Rs_rand)))
        #C1_0 = exp(logC1_0)
        #C1_0 = 1 / self.mem.cdf_R(self.R_RoI) / Omega
        # C1 = 1 / Omega / self.center0.distance.pc**2 / (np.mean(mem.density_2d(_Rs)))
        #C1 = C1_0 * C0
        #C1 = 1 / self.mem.cdf_R(self.R_RoI)
        #logC1 = logC1_0 + logC0
        #s = C0/(sigmafg*mem.density_2d(self.Rs)*C1 + C0)
        #s = C0/(sigmafg*mem.density_2d(self.Rs)*C1)
        #s = 1/(1+ 1/odds * exp(logC1_0+mem.logdensity_2d(ref_R)))
        s = 1/(1+ 1/odds * mem.density_2d_truncated(ref_R,self.R_RoI)/fg.density_2d_truncated(ref_R,self.R_RoI))
        #print("mem.density2d:",mem.density_2d(ref_R))
        #print("mem.mean_density2d:",mem.mean_density_2d(self.R_RoI))
        
        _Rs = (self.Rs(dra0,dde0) if Rs is None else Rs)
        #ret = log(C0) + log(s*C1_0*mem.density_2d(Rs)+(1-s))
        logfmem = log(mem.density_2d_truncated(_Rs,self.R_RoI))#logC1+mem.logdensity_2d(_Rs)
        logffg  = log( fg.density_2d_truncated(_Rs,self.R_RoI))#logC0*np.ones(shape=_Rs.shape)
        logfs = array([logfmem,logffg])
        ss = array([s,(1-s)])
        #print(logfmem.shape,logffg.shape,logfs.shape)
        logftot = logsumexp(a=np.transpose(logfs),b=ss,axis=1) # Note: (stars,components)*(components,), summed along with components axis 
        
        return (logfmem,logffg,logftot,s)
    
    def mem_cdf(R,dra0,dde0):
        R_photo = 0
        pass
    
    def __loglikelis(self,re,odds,dra0,dde0,Rs=None): # old fashion. do not use!
        mem = self.mem
        mem.update({"re_pc":re})
        ref_R = self.mem.half_light_radius() # 1.67834699001666*re for sersic model
        
        # C1 and C0 are the normalization factor of the member and foreground distribution.
        # For general area case, which is given by
        # \int_Omega d\Omega ~ Omega * (mean of p(R_i), where R_i denotes each samplinig points) 
        Omega = self.df_sample_config.loc["area"]
        Rs_rand = self._Rs(sc=self.sc_sample,dra0=dra0,dde0=dde0)
        C0 = 1 / Omega / self.center0.distance.pc**2
        C1_0 = 1 / np.mean(mem.density_2d(Rs_rand)) # C1/C0
        # C1 = 1 / Omega / self.center0.distance.pc**2 / (np.mean(mem.density_2d(_Rs)))
        C1 = C1_0 * C0
        #### C1 = 1 / self.center0.distance.pc**2 / (mem.cdf_R())
        #s = C0/(sigmafg*mem.density_2d(self.Rs)*C1 + C0)
        #s = C0/(sigmafg*mem.density_2d(self.Rs)*C1)
        s = 1/(1+ 1/odds * C1_0*mem.density_2d(ref_R))
        #print("mem.density2d:",mem.density_2d(ref_R))
        #print("mem.mean_density2d:",mem.mean_density_2d(self.R_RoI))
        
        _Rs = (self.Rs(dra0,dde0) if Rs==None else Rs)
        ret = log(C0) + log(s*C1_0*mem.density_2d(_Rs)+(1-s))

        return ret

    def lnposterior_general(self,params):
        lnp = self.lnprior(*params)
        if lnp > -inf:
            lnl = self.loglikeli(*params) 
            #print((self.beta*lnl+lnp)/lnl)
            return (self.beta*lnl+lnp, lnl)
        else:
            return (-inf, nan)
