import numpy as np
import numba
import matplotlib
import matplotlib.pyplot as plt
import glob
import sys
from scipy.constants import parsec, physical_constants
from scipy.special import betainc, beta
from scipy.integrate import quad
from numpy import pi, log, power, sqrt
from .dequad import dequad
#from py_header.DSphFuncs import DRACO_DISTANCE

kg_eV = 1./physical_constants["electron volt-kilogram relationship"][0]
im_eV = 1./physical_constants["electron volt-inverse meter relationship"][0]
solar_mass_kg = 1.9884e30
C0 = (solar_mass_kg*kg_eV)**2*((1./parsec)*im_eV)**5
C1 = (1e9)**2 * (1e2*im_eV)**5
C_J = C0/C1

def Rroi_pc(dist):
    return dist*np.deg2rad(0.5)

def rho_Msunpc3(r_pc,rhos_Msunpc3,rs_pc,a,b,g): #return [Msun]
    x = r_pc/rs_pc
    return rhos_Msunpc3*power(x,-g)*power(1+np.power(x,a),-(b-g)/a)

def integrand_err_Jfactor(r_pc,rhos_Msunpc3,rs_pc,a,b,g,dist):
    x,x0 = r_pc/dist, Rroi_pc(dist)/dist
    return 4.*pi*rho_Msunpc3(r_pc,rhos_Msunpc3,rs_pc,a,b,g)**2 * x * log((sqrt(1-x0*x0)-sqrt(x*x-x0*x0))/(1-x))

def Jfactor_err(rhos_Msunpc3,rs_pc,a,b,g,dist,*,Rtrunc_pc=2000):
    integ, err = quad(integrand_err_Jfactor,Rroi_pc(dist),Rtrunc_pc,args=(rhos_Msunpc3,rs_pc,a,b,g,dist))
    return C_J * integ

def Jfactor_err_dequad(rhos_Msunpc3,rs_pc,a,b,g,dist,*,Rtrunc_pc=2000,width=5e-3,pN=1000,mN=1000,show_fig=True): 
    args = [arg[:,np.newaxis] for arg in [rhos_Msunpc3,rs_pc,a,b,g,dist]]
    func = lambda r_pc: (Rtrunc_pc-Rroi_pc(args[-1]))/2 * integrand_err_Jfactor((Rtrunc_pc-Rroi_pc(args[-1]))/2 * r_pc + (Rtrunc_pc+Rroi_pc(args[-1]))/2,*args)
    integ = dequad(func,a=-1,b=1,axis=1,ignore_nan=True,show_fig=show_fig,width=width,pN=pN,mN=mN)
    return C_J * integ

def Jfactor_v01(rhos_Msunpc3,rs_pc,a,b,g,dist):
    xa = np.power(Rroi_pc(dist)/rs_pc,a)
    argbeta0 = (3.-2.*g)/a
    argbeta1 = (2.*b-3.)/a
    argbeta2 = (5.-2.*g)/a
    argbeta3 = (5.*b-3.)/a
    return C_J * (
        (4.*np.pi*rs_pc**3/a)*rhos_Msunpc3**2*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,xa/(1+xa))/dist**2
        +(1./3.)*(4.*np.pi*rs_pc**5/a)*rhos_Msunpc3**2*beta(argbeta2,argbeta3)*betainc(argbeta2,argbeta3,xa/(1+xa))/dist**4
        ) #M_sun/pc^2

def Jfactor_v01_truncated(rhos_Msunpc3,rs_pc,a,b,g,dist,Rtrunc_pc=2000):
    xa = np.power(Rtrunc_pc/rs_pc,a)
    argbeta0 = (3.-2.*g)/a
    argbeta1 = (2.*b-3.)/a
    argbeta2 = (5.-2.*g)/a
    argbeta3 = (5.*b-3.)/a
    return C_J * (
        (4.*np.pi*rs_pc**3/a)*rhos_Msunpc3**2*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,xa/(1+xa))/dist**2
        +(1./3.)*(4.*np.pi*rs_pc**5/a)*rhos_Msunpc3**2*beta(argbeta2,argbeta3)*betainc(argbeta2,argbeta3,xa/(1+xa))/dist**4
        ) #M_sun/pc^2
  
#@numba.vectorize(['f8[:](f8[:],f8[:],f8[:],f8[:],f8[:])'])
#@numba.guvectorize('f8[:](f8[:],f8[:],f8[:],f8[:],f8[:],f8[:])','(m),(m),(m),(m),(m),(m)->(m)')
def Jfactor_v02(rhos_Msunpc3,rs_pc,a,b,g,dist,*,Rtrunc_pc=2000):
    if Rtrunc_pc < Rroi_pc(dist):
        xa = np.power(Rtrunc_pc/rs_pc,a)
        argbeta0 = (3.-2.*g)/a
        argbeta1 = (2.*b-3.)/a
        argbeta2 = (5.-2.*g)/a
        argbeta3 = (5.*b-3.)/a
        return C_J * (
            (4.*np.pi*rs_pc**3/a)*rhos_Msunpc3**2*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,xa/(1+xa))/dist**2
            +(1./3.)*(4.*np.pi*rs_pc**5/a)*rhos_Msunpc3**2*beta(argbeta2,argbeta3)*betainc(argbeta2,argbeta3,xa/(1+xa))/dist**4
            ) #M_sun/pc^2
    else:
        #for i in np.arange(rhos_Msunpc3.shape[0]):
        #    ret[i] = Jfactor_v01(rhos_Msunpc3[i],rs_pc[i],a[i],b[i],g[i]) + Jfactor_err(rhos_Msunpc3[i],rs_pc[i],a[i],b[i],g[i])
        return Jfactor_v01(rhos_Msunpc3,rs_pc,a,b,g,dist) + Jfactor_err(rhos_Msunpc3,rs_pc,a,b,g,dist=dist,Rtrunc_pc=Rtrunc_pc)
        
Jfactor_v02 = np.vectorize(Jfactor_v02)

def Jfactor(rhos_Msunpc3,rs_pc,a,b,g,dist,*,Rtrunc_pc=2000,return_relerr=False,width=5e-3,pN=1000,mN=1000,show_fig=False):
    ret = np.zeros_like(rhos_Msunpc3)
    is_truncated = Rtrunc_pc < Rroi_pc(dist)
    isnot_truncated = np.logical_not(is_truncated)
    args_truncated = [arg[is_truncated] for arg in [rhos_Msunpc3,rs_pc,a,b,g,dist]]
    args_not_truncated = [arg[isnot_truncated] for arg in [rhos_Msunpc3,rs_pc,a,b,g,dist]]
    
    ret[is_truncated]  = Jfactor_v01_truncated(*args_truncated,Rtrunc_pc)
    
    ret_isnot_truncated = Jfactor_v01(*args_not_truncated)
    ret_isnot_truncated_err = Jfactor_err_dequad(*args_not_truncated,Rtrunc_pc=Rtrunc_pc,width=width,pN=pN,mN=mN,show_fig=show_fig)
    ret[isnot_truncated] = ret_isnot_truncated + ret_isnot_truncated_err
    
    return (ret if (not return_relerr) else (ret, ret_isnot_truncated_err/ret_isnot_truncated) )


'''
print(np.log10(Jfactor_v01(np.power(10,-2.05),np.power(10,3.96),2.78,7.78,0.675)))
print(np.log10(Jfactor_v02(np.power(10,-2.05),np.power(10,3.96),2.78,7.78,0.675)))
print(np.log10(Jfactor_v02(np.power(10,-1.52),np.power(10,3.15),2.77,3.18,0.783)))
print(np.log10(Jfactor_v02(np.power(10,-0.497),np.power(10,2.60),1.64,5.29,0.777)))
quit()
'''

if __name__ == '__main__':

    argvs = sys.argv
    argc = len(argvs)
    if(argc<2):
        print('usage: ', argvs[0], '(plotted filename)')
        quit()
    logMCparams_filenames_list = [glob.glob(argv) for argv in argvs[1:]]
    logMCparams_list = np.array([np.loadtxt(logMCparams_filename,delimiter=',',comments='#')[0:-1,:] for logMCparams_filenames in logMCparams_filenames_list for logMCparams_filename in logMCparams_filenames])
    print(logMCparams_list)
    #MCparams = np.concatenate(MCparams_list, axis=0)[ERASE_STEPS:,:]
    logMCparams = np.concatenate(logMCparams_list, axis=0)[ERASE_STEPS:,1:]
    ploted_step = logMCparams.shape[0]
    print('ploted file: ',logMCparams_filenames_list)
    print('ploted steps: ',ploted_step)
    '''
    logMCparams_list = np.array([np.loadtxt(argvs[i],delimiter=',',comments='#')[0:-1,:] for i in np.arange(1,argc)])
    #MCparams = np.concatenate(MCparams_list, axis=0)[ERASE_STEPS:,:]
    logMCparams = np.concatenate(logMCparams_list, axis=0)[ERASE_STEPS:,1:]
    ploted_step = logMCparams.shape[0]
    print('plot: ', argvs[1:])
    '''
    print(logparams_name[1:6])
    log10_rhos_Msunpc3,log10_rs_pc,a,b,g = logMCparams[:,1:6].T
    rhos_Msunpc3 = np.power(10,log10_rhos_Msunpc3)
    rs_pc = np.power(10,log10_rs_pc)
    #print(log10_rhos_Msunpc3,log10_rs_pc,a,b,g)
    
    Jfactor_ = Jfactor_v02(rhos_Msunpc3,rs_pc,a,b,g)
    num = Jfactor_.size
    print(num)
    ind_lo = int(num*0.16)
    ind_hi = int(num*0.84)
    log10Jfactor = np.log10(Jfactor_)
    log10Jfactor_sorted = np.sort(log10Jfactor)
    log10Jfacotr_mean = np.mean(log10Jfactor)
    log10Jfacotr_median = np.median(log10Jfactor)
    log10Jgactor_lo = log10Jfactor_sorted[ind_lo]
    log10Jfactor_up = log10Jfactor_sorted[ind_hi]
    
    print("Sample size: ",log10Jfactor.size,log10Jfactor)
    print('log10Jfacotr_mean:',log10Jfacotr_mean)
    print('log10Jfacotr_median:',log10Jfacotr_median)
    print('log10Jgactor_lo: ',log10Jgactor_lo)
    print('log10Jgactor_up:',log10Jfactor_up)
    #print("1sigma_lower:",log10Jfactor_lo)
    #print("1sigma_higher:",log10Jfactor_hi)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(log10Jfactor,bins=32)
    ax.set_xlabel(r'$\log_{10}(J/[GeV^2/cm^5])$')
    ax.set_ylabel('counts')
    fig.show()
    
    input("press any key")