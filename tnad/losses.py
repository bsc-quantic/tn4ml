import math
import quimb

def loss_miss(phi, P):
    phi_orig_renamed = phi.reindex({f'k{site}':f'k_{site}' for site in range(phi.nsites)})
    P_orig_renamed = P.reindex({f'k{site}':f'k_{site}' for site in range(P.nsites)})
    print((phi_orig_renamed.H&P_orig_renamed.H&P&phi)^all)
    # new_x = jnp.array(x, float)
    return math.pow((math.log((phi_orig_renamed.H&P_orig_renamed.H&P&phi)^all) - 1), 2)

def loss_reg(P, alpha):
    return alpha*max(0, math.log((P.H&P)^all))

def loss_miss_wrapped(P,phi=None,coeff=1):
    if phi==None:
        print("You can't use this function without a phi!")
        raise ValueError
    loss = loss_miss(phi,P)
    return coeff * loss

def loss_reg_wrapped(P,alpha=0.4):
    return loss_reg(P,alpha)
