import jax
import jax.numpy as jnp
from jax import lax

def eval_legendre_scalar(n, x, dtype=jnp.float64):
    """Helper function for scalar n value"""
    x = jnp.asarray(x, dtype=dtype)

    def body_fn(i, vals):
        P_nm1, P_n = vals
        P_np1 = ((2 * i + 1) * x * P_n - i * P_nm1) / (i + 1)
        return P_n, P_np1

    P0 = jnp.ones_like(x)
    P1 = x

    return lax.cond(
        n == 0,
        lambda _: P0,
        lambda _: lax.cond(
            n == 1,
            lambda _: P1,
            lambda _: lax.fori_loop(1, n, body_fn, (P0, P1))[1],
            operand=None
        ),
        operand=None
    )

def eval_legendre(n, x, dtype=jnp.float64):
    """
    Evaluates the Legendre polynomial of degree n at points x.
    
    Parameters
    ----------
    n : int or array-like of ints
        Degree(s) of the polynomial.
    x : float or array-like
        Point(s) at which to evaluate.
    dtype : jax.numpy.dtype, optional
        Data type of the output.
    
    Returns
    -------
        float or array-like: P_n(x)
    """
    # Check if n is an array
    try:
        n_shape = jnp.shape(n)
        is_array = len(n_shape) > 0
    except:
        is_array = False
    
    if is_array:
        # Vectorize over n using vmap
        return jax.vmap(lambda n_i: eval_legendre_scalar(n_i, x, dtype))(jnp.asarray(n))
    else:
        return eval_legendre_scalar(n, x, dtype)

def eval_laguerre_scalar(n, x, dtype=jnp.float64):
    """Helper function for scalar n value"""
    x = jnp.asarray(x, dtype=dtype)

    def body_fn(i, vals):
        L_nm2, L_nm1 = vals
        L_n = ((2 * i - 1 - x) * L_nm1 - (i - 1) * L_nm2) / i
        return L_nm1, L_n

    L0 = jnp.ones_like(x)
    L1 = 1.0 - x

    return lax.cond(
        n == 0,
        lambda _: L0,
        lambda _: lax.cond(
            n == 1,
            lambda _: L1,
            lambda _: lax.fori_loop(2, n + 1, body_fn, (L0, L1))[1],
            operand=None
        ),
        operand=None
    )

def eval_laguerre(n, x, dtype=jnp.float64):
    """
    Evaluates the Laguerre polynomial of degree n at points x.

    Parameters
    ----------
    n : int or array-like of ints
        Degree(s) of the polynomial.
    x : float or array-like
        Point(s) at which to evaluate.
    dtype : jax.numpy.dtype, optional
        Data type of the output.

    Returns
    -------
        float or array-like: L_n(x)
    """
    # Check if n is an array
    try:
        n_shape = jnp.shape(n)
        is_array = len(n_shape) > 0
    except:
        is_array = False
    
    if is_array:
        # Vectorize over n using vmap
        return jax.vmap(lambda n_i: eval_laguerre_scalar(n_i, x, dtype))(jnp.asarray(n))
    else:
        return eval_laguerre_scalar(n, x, dtype)

def eval_hermite_scalar(n, x, dtype=jnp.float64):
    """Helper function for scalar n value"""
    x = jnp.asarray(x, dtype=dtype)

    def body_fn(i, vals):
        H_nm2, H_nm1 = vals
        H_n = 2 * x * H_nm1 - 2 * (i - 1) * H_nm2
        return H_nm1, H_n

    H0 = jnp.ones_like(x)
    H1 = 2.0 * x

    return lax.cond(
        n == 0,
        lambda _: H0,
        lambda _: lax.cond(
            n == 1,
            lambda _: H1,
            lambda _: lax.fori_loop(2, n + 1, body_fn, (H0, H1))[1],
            operand=None
        ),
        operand=None
    )

def eval_hermite(n, x, dtype=jnp.float64):
    """
    Evaluates the physicist's Hermite polynomial H_n(x).

    Parameters
    ----------
    n : int or array-like of ints
        Degree(s) of the polynomial.
    x : float or array-like
        Points at which to evaluate.
    dtype : jax.numpy.dtype, optional
        Data type of the output.

    Returns
    -------
        float or array-like: H_n(x)
    """
    # Check if n is an array
    try:
        n_shape = jnp.shape(n)
        is_array = len(n_shape) > 0
    except:
        is_array = False
    
    if is_array:
        # Vectorize over n using vmap
        return jax.vmap(lambda n_i: eval_hermite_scalar(n_i, x, dtype))(jnp.asarray(n))
    else:
        return eval_hermite_scalar(n, x, dtype)