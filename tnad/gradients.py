import math
import quimb

def gradient_miss(phi, P_orig, P_rem, sites):
    
    # local update with 2 sites
    if len(sites) == 2:
        if (sites[1] == P_orig.nsites-1 and sites[1] > sites[0]): idx_remove_right = None
        elif (sites[1] > sites[0]): idx_remove_right = sites[1]
        else: idx_remove_right = sites[0]

        if sites[0] == 0 and sites[1] > sites[0]: idx_remove_left = None
        elif sites[1] > sites[0]: idx_remove_left = sites[0]-1
        else: idx_remove_left = sites[1]-1
    elif len(sites) == 1: # global update per site
        if sites[0] == 0:
            idx_remove_right = sites[0]
            idx_remove_left = None
        elif sites[0] == P_orig.nsites-1:
            idx_remove_right = None
            idx_remove_left = sites[0]-1
        else:
            idx_remove_right = sites[0]
            idx_remove_left = sites[0]-1
    
    # relabel (quimb requirement)
    phi_orig_renamed = phi.reindex({f'k{site}':f'k_{site}' for site in range(phi.nsites)})
    P_orig_renamed = P_orig.reindex({f'k{site}':f'k_{site}' for site in range(P_orig.nsites)})
    # l2_norm
    l2_norm = (phi_orig_renamed.H&P_orig_renamed.H&P_orig&phi)^all
    
    # relabel (quimb requirement)
    to_reindex = dict()
    if idx_remove_right != None: to_reindex.update({f'bond_{idx_remove_right}': f'bond{idx_remove_right}'})
    if idx_remove_left != None: to_reindex.update({f'bond_{idx_remove_left}': f'bond{idx_remove_left}'})
    P_orig_renamed = P_orig_renamed.reindex(to_reindex)
    
    first = (phi.H&P_rem.H&P_orig_renamed&phi_orig_renamed)^all
    second = (phi_orig_renamed.H&P_orig_renamed.H&P_rem&phi)^all
    fs = first+second
    # relabel back
    if idx_remove_right != None: fs = fs.reindex({f'bond{idx_remove_right}': f'bond_{idx_remove_right}'})
    if idx_remove_left != None: fs = fs.reindex({f'bond{idx_remove_left}': f'bond_{idx_remove_left}'})
    
    return 2*(math.log(l2_norm) - 1) * (1 / l2_norm) * fs

def gradient_reg(P_orig, P_rem, alpha, sites):
    frob_norm_sq = (P_orig.H&P_orig)^all
    
    # local update with 2 sites
    if len(sites) == 2:
        if (sites[1] == P_orig.nsites-1 and sites[1] > sites[0]): idx_remove_right = None
        elif (sites[1] > sites[0]): idx_remove_right = sites[1]
        else: idx_remove_right = sites[0]

        if sites[0] == 0 and sites[1] > sites[0]: idx_remove_left = None
        elif sites[1] > sites[0]: idx_remove_left = sites[0]-1
        else: idx_remove_left = sites[1]-1
    elif len(sites) == 1: # global update per site
        if sites[0] == 0:
            idx_remove_right = sites[0]
            idx_remove_left = None
        elif sites[0] == P_orig.nsites-1:
            idx_remove_right = None
            idx_remove_left = sites[0]-1
        else:
            idx_remove_right = sites[0]
            idx_remove_left = sites[0]-1
    
    # relabel (quimb requirement)
    to_reindex = dict()
    if idx_remove_right != None: to_reindex.update({f'bond_{idx_remove_right}': f'bond{idx_remove_right}'})
    if idx_remove_left != None: to_reindex.update({f'bond_{idx_remove_left}': f'bond{idx_remove_left}'})
    P_orig_renamed = P_orig.reindex(to_reindex)
    
    relu_part = ((((P_rem.H&P_orig_renamed)^all) + ((P_orig_renamed.H&P_rem)^all)) if frob_norm_sq >= 1 else 0)
    
    # relabel back
    if relu_part!=0:
        to_reindex_back=dict()
        if idx_remove_right != None: to_reindex_back.update({f'bond{idx_remove_right}': f'bond_{idx_remove_right}'})
        if idx_remove_left != None: to_reindex_back.update({f'bond{idx_remove_left}': f'bond_{idx_remove_left}'})
        relu_part = relu_part.reindex(to_reindex_back)
    return 2*alpha*(1/frob_norm_sq) * relu_part