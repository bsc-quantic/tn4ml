import random
import argparse
import numpy as np
import math
import quimb.tensor as qtn
import quimb as qu
import smpo
from tqdm import tqdm
#import embeddings as e
import FeatureMap as fm
import itertools
from matplotlib import pyplot as plt
import tensorflow as tf
from timeit import default_timer as timer


def load_mnist_train_data(train_size):
    (train_X, train_y), _ = tf.keras.datasets.mnist.load_data()
    # class = 1 -> inliers
    train_data = train_X[train_y==1][:train_size]
    np.random.shuffle(train_data)
    return train_data

def data_preprocessing(train_data, pool_size=(2,2), strides=(2,2), padding='valid'):
    data = []
    for sample in train_data:
        sample_tf = tf.constant(sample)
        sample_tf = tf.reshape(sample_tf, [1, 28, 28, 1])
        max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
           strides=strides, padding=padding)
        sample = max_pool_2d(sample_tf).numpy().reshape(14, 14)
        data.append(sample/255)
    return data

def loss_miss(phi, P):
    phi_orig_renamed = phi.reindex({f'k{site}':f'k_{site}' for site in range(phi.nsites)})
    P_orig_renamed = P.reindex({f'k{site}':f'k_{site}' for site in range(P.nsites)})
    return math.pow((math.log((phi_orig_renamed.H&P_orig_renamed.H&P&phi)^all) - 1), 2)

def gradient_miss(phi, P_orig, P_rem, sites, N_features):
    
    if (sites[1] == N_features-1 and sites[1] > sites[0]): idx_remove_right = None
    elif (sites[1] > sites[0]): idx_remove_right = sites[1]
    else: idx_remove_right = sites[0]
    
    if sites[0] == 0 and sites[1] > sites[0]: idx_remove_left = None
    elif sites[1] > sites[0]: idx_remove_left = sites[0]-1
    else: idx_remove_left = sites[1]-1
    
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

def loss_reg(P, alpha):
    return alpha*max(0, math.log((P.H&P)^all))

def gradient_reg(P_orig, P_rem, alpha, sites, N_features):
    frob_norm_sq = (P_orig.H&P_orig)^all
    if (sites[1] == N_features-1 and sites[1] > sites[0]): idx_remove_right = None
    elif (sites[1] > sites[0]): idx_remove_right = sites[1]
    else: idx_remove_right = sites[0]
    
    if sites[0] == 0 and sites[1] > sites[0]: idx_remove_left = None
    elif sites[1] > sites[0]: idx_remove_left = sites[0]-1
    else: idx_remove_left = sites[1]-1
    
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

def train_SMPO(data, spacing, n_epochs, alpha, lamda_init=2e-3, decay_rate=0.01, bond_dim=4, init_func='normal', scale=0.5, batch_size=32):
    
    train_data = np.array(data)
    N_features = train_data.shape[1]*train_data.shape[2]
    train_data_batched = np.array(np.split(train_data, batch_size))
    n_iters = int(train_data.shape[0]/batch_size)
    
    # initialize P
    P_orig = smpo.SpacedMatrixProductOperator.rand(n=N_features, spacing=spacing, bond_dim=bond_dim, init_func=init_func, scale=scale)
    P = P_orig.copy(deep=True)

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
                for sample in train_data_batched[it]:
                    # create MPS for input sample
                    
                    phi, _ = fm.embed(sample.flatten(), fm.trigonometric)

                    #calculate loss
                    loss_miss_batch += loss_miss(phi, P_ref)

                    #calculate gradient
                    grad_miss += gradient_miss(phi, P_ref, P, sites, N_features)
                    
                # total loss
                loss = (1/batch_size)*(loss_miss_batch) + loss_reg(P_ref, alpha)
                #print(f'epoch: {epoch}, iteration: {it} -> loss={loss}')
                loss_array.append(loss)

                # gradient of loss miss
                grad_miss.drop_tags()
                grad_miss.add_tag(site_tags[0]); grad_miss.add_tag(site_tags[1])
                # gradient of loss reg
                grad_regular = gradient_reg(P_ref, P, alpha, sites, N_features)
                if grad_regular != 0:
                    grad_regular.drop_tags()
                    grad_regular.add_tag(site_tags[0]); grad_regular.add_tag(site_tags[1])
                # total gradient
                total_grad = (1/batch_size)*grad_miss + grad_regular

                # update tensor
                if epoch < 20:
                    tensor_new = tensor_orig - lamda_init*total_grad
                else:
                    lamda = lamda_init*math.pow((1 - decay_rate/100),epoch)
                    tensor_new = tensor_orig - lamda*total_grad

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

                    
                    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Read arguments for TN optimization')
    # train params
    parser.add_argument('-lamda_init', dest='lamda_init', type=float, default=2.e-3, help='Lamda init')
    parser.add_argument('-decay_rate', dest='decay_rate', type=float, default=0.01, help='Decay rate for lamda')
    parser.add_argument('-train_size', dest='train_size',type=int, default=6016, help='Number of training samples')
    parser.add_argument('-n_epochs', dest='n_epochs', type=int, help='Number of epochs')
    parser.add_argument('-batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    # loss param
    parser.add_argument('-alpha', dest='alpha', type=float, default=0.4, help='alpha parameter for regularization')
    # MPO params
    parser.add_argument('-spacing', dest='spacing', type=int, help='Spacing for SMPO')
    parser.add_argument('-bond_dim', dest='bond_dim',type=int, default=4, help='Bond dimension for SMPO')
    parser.add_argument('-init_func', dest='init_func',type=str, default='normal', help='Bond dimension for SMPO')
    parser.add_argument('-scale_init_p', dest='scale_init_p',type=float, default=1.0, help='The width of the distribution')
    # max pooling for mnist
    parser.add_argument('-strides', dest='strides',type=int, nargs="+", help='Strides for max pooling of image')
    parser.add_argument('-padding', dest='padding',type=str, default='valid', help='Padding for max pooling of image')
    parser.add_argument('-pool_size', dest='pool_size',type=int, nargs="+", help='Pool size for max pooling of image')
    # save
    parser.add_argument('-save_name_smpo', dest='save_name_smpo',type=str, help='Save SMPO model')
    parser.add_argument('-save_name_loss', dest='save_name_loss',type=str, help='Save loss values')


    args = parser.parse_args()
    
    # load training data
    train_data = load_mnist_train_data(train_size=args.train_size)
    data = data_preprocessing(train_data, strides=tuple(args.strides), pool_size=tuple(args.pool_size), padding=args.padding)
    
    # train SMPO model
    P, loss_array = train_SMPO(data, args.spacing, args.n_epochs, args.alpha, args.lamda_init, args.decay_rate, args.bond_dim, args.init_func, args.scale_init_p, args.batch_size)
    
    # save
    qu.save_to_disk(P, f'{args.save_name_smpo}.pkl')
    np.save(f'{args.save_name_loss}.npy', loss_array)