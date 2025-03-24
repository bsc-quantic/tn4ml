import jax.numpy as jnp
import numpy as np
from tn4ml.scipy.special import eval_laguerre, eval_hermite

def test_laguerre_polynomials():
    """Test Laguerre polynomials against known values"""
    
    # Test 1: Basic known values
    # L₀(x) = 1 for any x
    assert jnp.allclose(eval_laguerre(0, 1.5), 1.0)
    assert jnp.allclose(eval_laguerre(0, 0.0), 1.0)
    
    # L₁(x) = 1-x
    assert jnp.allclose(eval_laguerre(1, 2.0), -1.0)  # 1-2 = -1
    assert jnp.allclose(eval_laguerre(1, 0.5), 0.5)   # 1-0.5 = 0.5
    
    # L₂(x) = (x² - 4x + 2) / 2
    x = 2.0
    expected = (x**2 - 4*x + 2)/2  # = (4 - 8 + 2)/2 = -1
    assert jnp.allclose(eval_laguerre(2, x), expected)
    
    # L₃(x) = (-x³ + 9x² - 18x + 6) / 6
    x = 3.0
    expected = (-x**3 + 9*x**2 - 18*x + 6)/6  # = (-27 + 81 - 54 + 6)/6 = 6/6 = 1
    assert jnp.allclose(eval_laguerre(3, x), expected)
    
    # Test 2: Array input for x
    x_values = jnp.array([0.0, 1.0, 2.0])
    expected = 1.0 - x_values  # L₁(x) = 1-x
    assert jnp.allclose(eval_laguerre(1, x_values), expected)
    
    # Test 3: Array input for n
    n_values = jnp.array([0, 1, 2])
    x = 1.0
    expected = jnp.array([
        1.0,           # L₀(1) = 1
        0.0,           # L₁(1) = 1-1 = 0
        -0.5           # L₂(1) = (1 - 4 + 2)/2 = -0.5
    ])
    assert jnp.allclose(eval_laguerre(n_values, x), expected)
    
    # Test 4: Verify recurrence relation
    # L_{n}(x) = ((2n-1-x)L_{n-1}(x) - (n-1)L_{n-2}(x))/n
    x = 2.5
    n = 4
    L_n_minus_2 = eval_laguerre(n-2, x)  # L₂(x)
    L_n_minus_1 = eval_laguerre(n-1, x)  # L₃(x)
    L_n = ((2*n-1-x)*L_n_minus_1 - (n-1)*L_n_minus_2)/n
    assert jnp.allclose(eval_laguerre(n, x), L_n)
    
    print("All Laguerre polynomial tests passed!")

def test_hermite_polynomials():
    """Test physicist's Hermite polynomials against known values"""
    
    # Test 1: Basic known values
    # H₀(x) = 1 for any x
    assert jnp.allclose(eval_hermite(0, 1.5), 1.0)
    assert jnp.allclose(eval_hermite(0, 0.0), 1.0)
    
    # H₁(x) = 2x
    assert jnp.allclose(eval_hermite(1, 2.0), 4.0)    # 2*2 = 4
    assert jnp.allclose(eval_hermite(1, -1.5), -3.0)  # 2*(-1.5) = -3
    
    # H₂(x) = 4x² - 2
    x = 2.0
    expected = 4*x**2 - 2  # = 4*4 - 2 = 16 - 2 = 14
    assert jnp.allclose(eval_hermite(2, x), expected)
    
    # H₃(x) = 8x³ - 12x
    x = 1.5
    expected = 8*x**3 - 12*x  # = 8*3.375 - 12*1.5 = 27 - 18 = 9
    assert jnp.allclose(eval_hermite(3, x), expected)
    
    # H₄(x) = 16x⁴ - 48x² + 12
    x = -1.0
    expected = 16*x**4 - 48*x**2 + 12  # = 16*1 - 48*1 + 12 = 16 - 48 + 12 = -20
    assert jnp.allclose(eval_hermite(4, x), expected)
    
    # Test 2: Array input for x
    x_values = jnp.array([0.0, 0.5, 1.0])
    expected = 2.0 * x_values  # H₁(x) = 2x
    assert jnp.allclose(eval_hermite(1, x_values), expected)
    
    # Test 3: Array input for n
    n_values = jnp.array([0, 1, 2, 3])
    x = 0.5
    expected = jnp.array([
        1.0,           # H₀(0.5) = 1
        1.0,           # H₁(0.5) = 2*0.5 = 1
        -1.0,          # H₂(0.5) = 4*(0.5)² - 2 = 4*0.25 - 2 = 1 - 2 = -1
        -5.0           # H₃(0.5) = 8*(0.5)³ - 12*0.5 = 8*0.125 - 12*0.5 = 1 - 6 = -5
    ])
    assert jnp.allclose(eval_hermite(n_values, x), expected)
    
    # Test 4: Verify recurrence relation
    # H_n(x) = 2x·H_{n-1}(x) - 2(n-1)·H_{n-2}(x)
    x = 2.5
    n = 5
    H_n_minus_2 = eval_hermite(n-2, x)  # H₃(x)
    H_n_minus_1 = eval_hermite(n-1, x)  # H₄(x)
    H_n = 2*x*H_n_minus_1 - 2*(n-1)*H_n_minus_2
    assert jnp.allclose(eval_hermite(n, x), H_n)
    
    # Test 5: Special case - orthogonality at x=0
    # H_n(0) = 0 for odd n, and H_n(0) = (-1)^(n/2)·2^n·(n/2)! for even n
    assert jnp.allclose(eval_hermite(1, 0.0), 0.0)
    assert jnp.allclose(eval_hermite(3, 0.0), 0.0)
    assert jnp.allclose(eval_hermite(5, 0.0), 0.0)
    assert jnp.allclose(eval_hermite(2, 0.0), -2.0)  # (-1)¹·2²·1! = -1·4·1 = -4
    assert jnp.allclose(eval_hermite(4, 0.0), 12.0)  # (-1)²·2⁴·2! = 1·16·2 = 32
    
    print("All Hermite polynomial tests passed!")