#!/usr/bin/env python3
"""
Test script for model_jax.py
"""

import jax.numpy as np
import jax
from jeanspy.model_jax import *

def test_plummer_model():
    print("Testing PlummerModel...")
    model = PlummerModel(re_pc=100.0)
    
    R = np.linspace(0.1, 1000.0, 100)
    density_2d = model.density_2d(R)
    density_3d = model.density_3d(R)
    cdf = model.cdf_R(R)
    
    print(f"  Plummer model created with re_pc={model.re_pc}")
    print(f"  density_2d at R=100pc: {density_2d[50]:.6e}")
    print(f"  density_3d at r=100pc: {density_3d[50]:.6e}")
    print(f"  cdf at R=100pc: {cdf[50]:.6f}")
    print(f"  half_light_radius: {model.half_light_radius()}")

def test_sersic_model():
    print("\nTesting SersicModel...")
    model = SersicModel(re_pc=200.0, n=2.0)
    
    R = np.linspace(0.1, 1000.0, 100)
    density_2d = model.density_2d(R)
    density_3d = model.density_3d(R)
    cdf = model.cdf_R(R)
    
    print(f"  Sersic model created with re_pc={model.re_pc}, n={model.n}")
    print(f"  b parameter: {model.b:.6f}")
    print(f"  density_2d at R=200pc: {density_2d[50]:.6e}")
    print(f"  density_3d at r=200pc: {density_3d[50]:.6e}")
    print(f"  cdf at R=200pc: {cdf[50]:.6f}")

def test_exp2d_model():
    print("\nTesting Exp2dModel...")
    model = Exp2dModel(re_pc=150.0)
    
    R = np.linspace(0.1, 1000.0, 100)
    density_2d = model.density_2d(R)
    density_3d = model.density_3d(R)
    cdf = model.cdf_R(R)
    
    print(f"  Exp2d model created with re_pc={model.re_pc}")
    print(f"  R_exp_pc: {model.R_exp_pc:.6f}")
    print(f"  density_2d at R=150pc: {density_2d[50]:.6e}")
    print(f"  density_3d at r=150pc: {density_3d[50]:.6e}")
    print(f"  cdf at R=150pc: {cdf[50]:.6f}")

def test_nfw_model():
    print("\nTesting NFWModel...")
    model = NFWModel(rs_pc=1000.0, rhos_Msunpc3=1e6, r_t_pc=5000.0)
    
    r = np.linspace(0.1, 5000.0, 100)
    density_3d = model.mass_density_3d(r)
    mass = model.enclosure_mass(r)
    
    print(f"  NFW model created with rs_pc={model.rs_pc}, rhos_Msunpc3={model.rhos_Msunpc3}")
    print(f"  density_3d at r=1000pc: {density_3d[50]:.6e}")
    print(f"  enclosure_mass at r=1000pc: {mass[50]:.6e}")

def test_zhao_model():
    print("\nTesting ZhaoModel...")
    model = ZhaoModel(rs_pc=1000.0, rhos_Msunpc3=1e6, a=1.0, b=3.0, g=1.0, r_t_pc=5000.0)
    
    r = np.linspace(0.1, 5000.0, 100)
    density_3d = model.mass_density_3d(r)
    mass = model.enclosure_mass(r)
    
    print(f"  Zhao model created with rs_pc={model.rs_pc}, a={model.a}, b={model.b}, g={model.g}")
    print(f"  density_3d at r=1000pc: {density_3d[50]:.6e}")
    print(f"  enclosure_mass at r=1000pc: {mass[50]:.6e}")

def test_anisotropy_models():
    print("\nTesting Anisotropy Models...")
    
    # Constant anisotropy
    const_model = ConstantAnisotropyModel(beta_const=0.2)
    r = np.linspace(100.0, 1000.0, 10)
    beta_const = const_model.beta(r)
    print(f"  Constant anisotropy β={const_model.beta_const}: {beta_const[0]:.3f}")
    
    # Osipkov-Merritt
    om_model = OsipkovMerrittModel(r_a=500.0)
    beta_om = om_model.beta(r)
    print(f"  Osipkov-Merritt at r=500pc: {beta_om[4]:.3f}")
    
    # Baes
    baes_model = BaesAnisotropyModel(beta_0=0.0, beta_inf=0.7, r_a=300.0, eta=2.0)
    beta_baes = baes_model.beta(r)
    print(f"  Baes model at r=500pc: {beta_baes[4]:.3f}")

def test_jax_functionality():
    print("\nTesting JAX functionality...")
    
    # Test JAX compilation
    model = PlummerModel(re_pc=100.0)
    
    @jax.jit
    def compiled_density(R):
        return model.density_2d(R)
    
    R = np.array([50.0, 100.0, 200.0])
    density_normal = model.density_2d(R)
    density_compiled = compiled_density(R)
    
    print(f"  Normal calculation: {density_normal}")
    print(f"  JIT compiled calculation: {density_compiled}")
    print(f"  Results match: {np.allclose(density_normal, density_compiled)}")
    
    # Test gradient
    @jax.grad
    def loss_fn(re_pc):
        model_grad = PlummerModel(re_pc=re_pc)
        R = np.array([100.0])
        density = model_grad.density_2d(R)
        return np.sum(density**2)
    
    gradient = loss_fn(100.0)
    print(f"  Gradient w.r.t. re_pc: {gradient:.6e}")

if __name__ == "__main__":
    print("Testing JAX models...")
    
    try:
        test_plummer_model()
        test_sersic_model()
        test_exp2d_model()
        test_nfw_model()
        test_zhao_model()
        test_anisotropy_models()
        test_jax_functionality()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
