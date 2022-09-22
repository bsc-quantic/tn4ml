import math
import quimb
from autoray import do
from autoray import numpy as np
from jax import numpy as jnp
from autoray import lazy

def loss_miss(P,phi=None,coeff=1,backend=None):
    if phi==None:
        print("You can't use this function without a phi!")
        raise ValueError
    phi_orig_renamed = phi.reindex({f'k{site}':f'k_{site}' for site in range(phi.nsites)})
    P_orig_renamed = P.reindex({f'k{site}':f'k_{site}' for site in range(P.nsites)})
    loss = do('power',do('log',((phi_orig_renamed.H&P_orig_renamed.H&P&phi)^all)) - 1, 2)

    return coeff * loss

def loss_reg(P,alpha=0.4,backend=None):
    return alpha*do('max',do('array',[0,do('log',(P.H&P)^all)],like=backend))