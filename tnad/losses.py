import math
import quimb

def loss_miss(phi, P):
    phi_orig_renamed = phi.reindex({f'k{site}':f'k_{site}' for site in range(phi.nsites)})
    P_orig_renamed = P.reindex({f'k{site}':f'k_{site}' for site in range(P.nsites)})
    return math.pow((math.log((phi_orig_renamed.H&P_orig_renamed.H&P&phi)^all) - 1), 2)

def loss_reg(P, alpha):
    return alpha*max(0, math.log((P.H&P)^all))