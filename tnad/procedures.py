import itertools
import numpy as np
import tnad.FeatureMap as fm
from tnad.losses import loss_miss, loss_reg
from tnad.gradients import gradient_miss, gradient_reg
import math
import quimb.tensor as qtn
import quimb as qu

def local_update_sweep_dyncanonization_renorm(P, n_epochs, n_iters, data, batch_size, alpha, lamda_init, bond_dim, decay_rate=None, expdecay_tol=None):
    N_features = P.nsites
    
    loss_array = []
    for epoch in range(n_epochs):
        for it in range(n_iters):
            # define sweeps
            sweeps = itertools.chain(zip(list(range(0,N_features-1)), list(range(1,N_features))), reversed(list(zip(list(range(1,N_features)),list(range(0,N_features-1))))))
            for sweep_it, sites in enumerate(sweeps):
                [sitel, siter] = sites
                site_tags = [P.site_tag(site) for site in sites]
                # canonize P with root in sites
                ortog_center = sites
                P.canonize(sites, cur_orthog=ortog_center)
                # copy P as reference
                P_ref = P.copy(deep=True)
                # pop site tensor
                [origl, origr] = P.select_tensors(site_tags, which="any")
                tensor_orig = origl & origr ^ all
                # memorize bond between 2 selected sites
                bond_ind_removed = P.bond(site_tags[0], site_tags[1])

                #virtual bonds
                #    left
                if sitel == 0 or (sitel == N_features-1 and sitel>siter): vindl = []
                elif sitel>0 and sitel<siter: vindl = [P.bond(sitel-1, sitel)]
                else: vindl = [P.bond(sitel, sitel+1)]
                #    right
                if siter == N_features - 1 or (siter == 0 and siter<sitel): vindr = []
                elif siter < N_features-1 and siter>sitel: vindr = [P.bond(siter, siter+1)]
                else: vindr = [P.bond(siter-1, siter)]

                # remove site tags of poped sites
                P.delete(site_tags, which="any")

                grad_miss=0; loss_miss_batch=0
                for sample in data[it]:
                    # create MPS for input sample
                    phi, _ = fm.embed(sample.flatten(), fm.trigonometric)
                    
                    #calculate loss
                    loss_miss_batch += loss_miss(phi, P_ref)
                    
                    #calculate gradient
                    grad_miss += gradient_miss(phi, P_ref, P, sites)
                # total loss
                loss = (1/batch_size)*(loss_miss_batch)
                loss_array.append(loss)

                # gradient of loss miss
                grad_miss.drop_tags()
                grad_miss.add_tag(site_tags[0]); grad_miss.add_tag(site_tags[1])
                # gradient of loss reg
                # grad_regular = gradient_reg(P_ref, P, alpha, sites, N_features)
                # if grad_regular != 0:
                #     grad_regular.drop_tags()
                #     grad_regular.add_tag(site_tags[0]); grad_regular.add_tag(site_tags[1])
                # total gradient
                total_grad = (1/batch_size)*grad_miss

                # update tensor
                if decay_rate != None:
                    if epoch > expdecay_tol:
                        # exp. decay of lamda
                        lamda = lamda_init*math.pow((1 - decay_rate/100),epoch)
                        tensor_new = tensor_orig - lamda*total_grad
                else:
                    tensor_new = tensor_orig - lamda_init*total_grad

                # normalize updated tensor
                tensor_new.normalize(inplace=True)

                # split updated tensor in 2 tensors
                lower_ind = [f'b{sitel}'] if f'b{sitel}' in P.lower_inds else []
                [tensorl, tensorr] = tensor_new.split(get="tensors", left_inds=[*vindl, P.upper_ind(sitel), *lower_ind], bond_ind=bond_ind_removed, max_bond=bond_dim)

                # link new tensors to P back
                for site, tensor in zip(sites, [tensorl, tensorr]):
                    tensor.drop_tags()
                    tensor.add_tag(P.site_tag(site))
                    P.add_tensor(tensor)
    return P, loss_array

def get_sample_grad(sample, embed_func, P, P_rem, tensor):
    # create MPS for input sample
    phi, _ = fm.embed(sample.flatten(), embed_func)

    #calculate gradient
    grad_miss = gradient_miss(phi, P, P_rem, [tensor])
    return grad_miss

def get_sample_loss(sample, embed_func, P):
    # create MPS for input sample
    phi, _ = fm.embed(sample.flatten(), embed_func)

    #calculate loss
    loss_miss_batch = loss_miss(phi, P)
    return loss_miss_batch

def get_total_grad(P, tensor, data, embed_func, batch_size, alpha):
    P_rem = P.copy(deep=True)
    
    site_tag = P_rem.site_tag(tensor)
    # remove site tag of poped sites
    P_rem.delete(site_tag, which="any")

    # paralelize
    grad_miss = []
    for i, sample in enumerate(data):
        output_per_sample = get_sample_grad(sample, embed_func, P, P_rem, tensor)
        grad_miss.append(output_per_sample)
    
    # gradient of loss miss
    grad_miss = sum(grad_miss)
    grad_miss.drop_tags()
    grad_miss.add_tag(site_tag)
    # gradient of loss reg
    grad_regular = gradient_reg(P, P_rem, alpha, [tensor])
    if grad_regular != 0:
        grad_regular.drop_tags()
        grad_regular.add_tag(site_tag)
    # total gradient
    total_grad = (1/batch_size)*(grad_miss) + grad_regular
    return total_grad

def global_update_costfuncnorm(P, n_epochs, n_iters, data, batch_size, alpha, lamda_init, bond_dim, decay_rate=None, expdecay_tol=None):
    loss_array = []
    n_tensors = P.nsites
    
    for epoch in range(n_epochs):
        for it in range(n_iters):            
            # paralelize
            grad_per_tensor=[]
            for tensor in range(n_tensors):
                embed_func = fm.trigonometric
                output_per_tensor = get_total_grad(P, tensor, data[it], embed_func, batch_size, alpha) # get grad per tensor
                grad_per_tensor.append(output_per_tensor)
            
            # get loss per sample
            loss_miss = 0
            for i, sample in enumerate(data[it]):
                embed_func = fm.trigonometric
                output_per_sample = get_sample_loss(sample, embed_func, P)
                loss_miss += output_per_sample
            # get total loss
            total_loss = (1/batch_size)*(loss_miss) + loss_reg(P, alpha)
            loss_array.append(total_loss)

            # update P
            # no need to paralelize
            for tensor in range(n_tensors):
                site_tag = P.site_tag(tensor)
                tensor_orig = P.select_tensors(site_tag, which="any")
                
                if epoch > expdecay_tol:
                    if decay_rate != None:
                        # exp. decay of lamda
                        lamda = lamda_init*math.pow((1 - decay_rate/100),epoch)
                        tensor_orig = tensor_orig - lamda*grad_per_tensor[tensor]
                else:
                    tensor_orig = tensor_orig - lamda_init*grad_per_tensor[tensor]
                
    return P, loss_array