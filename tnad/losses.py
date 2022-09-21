import math
import quimb
import jax.numpy as jnp

def loss_miss(phi,P):
    phi_orig_renamed = phi.reindex({f'k{site}':f'k_{site}' for site in range(phi.nsites)})
    P_orig_renamed = P.reindex({f'k{site}':f'k_{site}' for site in range(P.nsites)})
    return math.pow((math.log((phi_orig_renamed.H&P_orig_renamed.H&P&phi)^all) - 1), 2)

def loss_reg(P, alpha):
    return alpha*max(0,math.log((P.H&P)^all))

# these are the versions of loss wrapped to work with jax

def loss_miss_wrapped(P,phi=None,coeff=1):
    if phi==None:
        print("You can't use this function without a phi!")
        raise ValueError
    phi_orig_renamed = phi.reindex({f'k{site}':f'k_{site}' for site in range(phi.nsites)})
    P_orig_renamed = P.reindex({f'k{site}':f'k_{site}' for site in range(P.nsites)})
    loss = jnp.power((jnp.log((phi_orig_renamed.H&P_orig_renamed.H&P&phi)^all) - 1), 2)

    return coeff * loss

def loss_reg_wrapped(P,alpha=0.4):
    return alpha*jnp.max(jnp.array([0, jnp.log((P.H&P)^all)]))