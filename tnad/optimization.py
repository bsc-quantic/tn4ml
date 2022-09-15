import argparse
import numpy as np
import math
import quimb.tensor as qtn
import quimb as qu
from tnad import smpo
#import embeddings as e
import tnad.FeatureMap as fm
import tnad.procedures as p
import itertools
from matplotlib import pyplot as plt
import tensorflow as tf
from time import time


def load_mnist_train_data(train_size, seed: int = None):
    (train_X, train_y), _ = tf.keras.datasets.mnist.load_data()
    # class = 1 -> inliers
    train_data = train_X[train_y==1][:train_size]
    
    if(seed != None):
        np.random.seed(seed)
    np.random.shuffle(train_data)
    return train_data

def data_preprocessing(train_data, pool_size=(2,2), strides=(2,2), padding='valid', reduced_shape=(14, 14)):
    data = []
    for sample in train_data:
        sample_tf = tf.constant(sample)
        sample_tf = tf.reshape(sample_tf, [1, 28, 28, 1])
        max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
           strides=strides, padding=padding)
        sample = max_pool_2d(sample_tf).numpy().reshape(reduced_shape)
        data.append(sample/255)
    return data

def train_SMPO(data, spacing, n_epochs, alpha, opt_procedure, lamda_init=2e-3, decay_rate=None, expdecay_tol=None, bond_dim=4, init_func='normal', scale=0.5, batch_size=32, seed: int = None):
    
    train_data = np.array(data)
    N_features = train_data.shape[1]*train_data.shape[2]
    train_data_batched = np.array(np.split(train_data, batch_size))
    n_iters = int(train_data.shape[0]/batch_size)
    
    # initialize P
    P_orig = smpo.SpacedMatrixProductOperator.rand(n=N_features, spacing=spacing, bond_dim=bond_dim, init_func=init_func, scale=scale, seed=seed)
    P = P_orig.copy(deep=True)
    
    P, loss_array = opt_procedure(P, n_epochs, n_iters, train_data_batched, batch_size, alpha, lamda_init, bond_dim, decay_rate=decay_rate, expdecay_tol=expdecay_tol)
    return P, loss_array

                    
                    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Read arguments for TN optimization')
    # train params
    parser.add_argument('-lamda_init', dest='lamda_init', type=float, default=2.e-3, help='Lamda init')
    parser.add_argument('-decay_rate', dest='decay_rate', type=float, help='Decay rate for lamda')
    parser.add_argument('-expdecay_tol', dest='expdecay_tol', type=int, help='Number of epochs before lamda starts to decay exponentialy')
    parser.add_argument('-train_size', dest='train_size',type=int, default=6016, help='Number of training samples')
    parser.add_argument('-n_epochs', dest='n_epochs', type=int, help='Number of epochs')
    parser.add_argument('-batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-opt_procedure', dest='opt_procedure', type=str, help='Optimization procedure')
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
    parser.add_argument('-reduced_shape', dest='reduced_shape',type=int, nargs="+", help='Reduced shape of image')
    # save
    parser.add_argument('-save_name_smpo', dest='save_name_smpo',type=str, help='Save SMPO model')
    parser.add_argument('-save_name_loss', dest='save_name_loss',type=str, help='Save loss values')


    args = parser.parse_args()
    
    # load training data
    train_data = load_mnist_train_data(train_size=args.train_size)
    data = data_preprocessing(train_data, strides=tuple(args.strides), pool_size=tuple(args.pool_size), padding=args.padding, reduced_shape=tuple(args.reduced_shape))
    
    # train SMPO model
    if args.opt_procedure == 'local_2sitesweep_dynamic_canonization_renorm':
        opt_procedure = p.local_2sitesweep_dynamic_canonization_renorm
    elif args.opt_procedure == 'global_update_costfuncnorm':
        opt_procedure = p.global_update_costfuncnorm
        
    P, loss_array = train_SMPO(data, args.spacing, args.n_epochs, args.alpha, opt_procedure, args.lamda_init, args.decay_rate, args.expdecay_tol, args.bond_dim, args.init_func, args.scale_init_p, args.batch_size)
    
    # save
    qu.save_to_disk(P, f'{args.save_name_smpo}.pkl')
    np.save(f'{args.save_name_loss}.npy', loss_array)
