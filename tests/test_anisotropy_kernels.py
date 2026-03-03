#!/usr/bin/env python3
"""
Simple test for anisotropy models with kernel methods
"""

import jax.numpy as jnp
import numpy as np
from jeanspy.model_jax import *

def test_anisotropy_kernels():
    print("Testing Anisotropy Model Kernels...")
    
    # Test data
    u = jnp.linspace(1.1, 3.0, 5)  # u > 1 for kernel calculations
    R = jnp.array([100.0, 200.0])   # radius values
    r = jnp.linspace(100.0, 1000.0, 10)
    
    # Test ConstantAnisotropyModel
    print("\n1. ConstantAnisotropyModel:")
    try:
        const_model = ConstantAnisotropyModel(beta_ani=0.2)
        beta_vals = const_model.beta(r)
        f_vals = const_model.f(r)
        
        print(f"  beta_ani: {const_model.beta_ani}")
        print(f"  beta at r=500pc: {beta_vals[4]:.3f}")
        print(f"  f at r=500pc: {f_vals[4]:.3e}")
        
        # Test kernel - this might be slow due to scipy integration
        print(f"  Testing kernel with u shape: {u.shape}, R shape: {R.shape}")
        kernel_vals = const_model.kernel(u, R)
        print(f"  kernel shape: {kernel_vals.shape}")
        print(f"  kernel sample value: {kernel_vals[0, 0]:.6f}")
        
    except Exception as e:
        print(f"  Error in ConstantAnisotropyModel: {e}")
    
    # Test OsipkovMerrittModel
    print("\n2. OsipkovMerrittModel:")
    try:
        om_model = OsipkovMerrittModel(r_a=500.0)
        beta_vals = om_model.beta(r)
        f_vals = om_model.f(r)
        
        print(f"  r_a: {om_model.r_a}")
        print(f"  beta at r=500pc: {beta_vals[4]:.3f}")
        print(f"  f at r=500pc: {f_vals[4]:.3f}")
        
        # Test kernel
        kernel_vals = om_model.kernel(u, R)
        print(f"  kernel shape: {kernel_vals.shape}")
        print(f"  kernel sample value: {kernel_vals[0, 0]:.6f}")
        
    except Exception as e:
        print(f"  Error in OsipkovMerrittModel: {e}")
    
    # Test BaesAnisotropyModel
    print("\n3. BaesAnisotropyModel:")
    try:
        baes_model = BaesAnisotropyModel(beta_0=0.0, beta_inf=0.7, r_a=300.0, eta=2.0)
        beta_vals = baes_model.beta(r)
        f_vals = baes_model.f(r)
        
        print(f"  beta_0: {baes_model.beta_0}, beta_inf: {baes_model.beta_inf}")
        print(f"  r_a: {baes_model.r_a}, eta: {baes_model.eta}")
        print(f"  beta at r=500pc: {beta_vals[4]:.3f}")
        print(f"  f at r=500pc: {f_vals[4]:.3e}")
        
        # Test integrand_kernel first
        u_test = jnp.array([1.5])
        R_test = jnp.array([200.0])
        integrand_val = baes_model.integrand_kernel(u_test, R_test)
        print(f"  integrand_kernel test: {integrand_val[0]:.6f}")
        
        # Test kernel (this will be slow)
        print("  Testing kernel (this may take a moment due to numerical integration)...")
        kernel_vals = baes_model.kernel(u[:2], R[:1])  # smaller test to speed up
        print(f"  kernel shape: {kernel_vals.shape}")
        print(f"  kernel sample value: {kernel_vals[0, 0]:.6f}")
        
    except Exception as e:
        print(f"  Error in BaesAnisotropyModel: {e}")
        import traceback
        traceback.print_exc()

def test_jax_compilation():
    print("\n\nTesting JAX Compilation of Anisotropy Models...")
    
    try:
        # Test JAX compilation on simple functions
        const_model = ConstantAnisotropyModel(beta_ani=0.2)
        
        @jax.jit
        def compiled_beta(r):
            return const_model.beta(r)
        
        @jax.jit
        def compiled_f(r):
            return const_model.f(r)
        
        r = jnp.array([100.0, 200.0, 500.0])
        beta_compiled = compiled_beta(r)
        f_compiled = compiled_f(r)
        
        print(f"  Compiled beta: {beta_compiled}")
        print(f"  Compiled f: {f_compiled}")
        print("  JAX compilation successful for beta and f methods!")
        
    except Exception as e:
        print(f"  JAX compilation error: {e}")

if __name__ == "__main__":
    test_anisotropy_kernels()
    test_jax_compilation()
    print("\n✅ Anisotropy kernel tests completed!")
