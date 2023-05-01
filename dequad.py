from numpy import sinh,cosh,exp,log,pi,arange,isnan,isinf,float64,float128
from functools import lru_cache
import numpy as np
import warnings

DEBUG = False
BUF_DEBUG = None

def hashable(x):
    try:
        hash(x)
        return True
    except TypeError:
        return False

# 関数を Memoize するデコレータ.
def memorize(callable):
    cache = {}
    def wrapper(*args, **kwargs):
        key = args + tuple(kwargs.items())
        if not hashable(key):
            return callable(*args, **kwargs)
        if key not in cache:
            cache[key] = callable(*args, **kwargs)
        return cache[key]
    return wrapper

#@lru_cache(maxsize = 1)
@memorize
def generate_x_w(a,b,n,xp=np):
    '''
    x = phi(t)
    w = phi'(t)
    
    if a and b is scalar, xs and ws has shape (n,)
    if a or b is an array (n_edge,), return has shape () 
    '''
    #print("generate_x_w in",a,b,width,mN,pN)
    pi2 = xp.pi/2
    
    if np.all(np.isfinite(a)) and np.all(np.isinf(b)):  # a < x < inf
        ts = np.linspace(-6.85,6.79,n)  # finite x, finite w, w!=0
        width = ts[1] - ts[0]
        ps = pi2*xp.sinh(ts)
        xs = xp.exp(ps) + a
        ws = width *pi2 *xp.cosh(ts)*xp.exp(ps)    
        
    elif np.all(np.isinf(a)) and np.all(np.isinf(b)):  # -inf < x < inf
        ts = np.linspace(-6.79,6.79,n)  # finite x, finite w, w!=0
        width = ts[1] - ts[0]
        ps = pi2*xp.sinh(ts)
        xs = xp.sinh(ps)
        ws = width *pi2 *xp.cosh(ts)*xp.cosh(ps)  
        
    elif np.all(np.isfinite(a)) and np.all(np.isfinite(b)):  # a < x < b
        ts = np.linspace(-6.10,6.10,n)  # finite x, finite w, w!=0
        width = ts[1] - ts[0]
        ps = pi2*xp.sinh(ts)
        xs = (b-a)/2*xp.tanh(ps) + (a+b)/2
        ws = width * (b-a)/2 *pi2 *xp.cosh(ts)/xp.cosh(ps)/xp.cosh(ps)
        
    return xs,ws


def dequad(func,a,b,n,
           axis=None,xp=np,
           replace_inf_to_zero=False,
           replace_nan_to_zero=False,
           verbose=False):
    '''
    func: func(ndarray_in) = ndarray_out
    axis: define the axis of ndarray_out to use integrate.
    
    Note: If "func" is like func: array(n,) -> array(m,n),
              f(x) * w ~ (m,n) * (n,) ~ (m,n) * (1,n)
          then works well (axis should set as axis=1). 
          
          Similarly, funcions like: array(n,) -> array(m1,m2,...,n),
              f(x) * w ~ (m1,m2,...,n) * (n,) ~ (m1,m2,...,n) * (1,1,...,n)
          then works well (axis should set to be last axis). 
    '''
    xs,ws = generate_x_w(a,b,n,xp)
    wsfs = ws*func(xs)
     
    if replace_inf_to_zero:
        wsfs[np.isinf(wsfs)] = 0
    elif np.any(np.isinf(wsfs)):
        warnings.warn("inf is detected in dequad calculation. Use \"replace_inf_to_zero\" to ignore such elements")
        
    if replace_nan_to_zero:
        wsfs[np.isnan(wsfs)] = 0
    elif np.any(np.isnan(wsfs)):
        warnings.warn("nan is detected in dequad calculation. Use \"replace_nan_to_zero\" to ignore such elements")
        
    if verbose:    
        print(wsfs)
        
    return (wsfs).sum(axis=axis)
    

if __name__ == "__main__":
    #### demonstration and illustration ####
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def debug(func,a,b):
        ns = range(1,20)
        errs = [np.abs(1-dequad(func,a,b,n=2**i)) for i in ns]
        print(errs)
        plt.plot(list(ns),errs)
        plt.yscale("log")
        
    debug(lambda x: 1/np.sqrt(2*np.pi)*np.exp(-x**2/2),np.inf,np.inf)
    debug(lambda x: 1/x**2,1,np.inf)
    debug(lambda x: 2*np.exp(-x)*np.sin(x),0,np.inf)
    debug(lambda x: 2*x,0,1)
    
    plt.show()  # The plotted figure shows that n = 2**6~2**10 (64~1024) is good for the normal use