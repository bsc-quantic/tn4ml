# args
train_size = 1024
batch_size = 32
strides = (2,2)
pool_size = (2,2)
padding = 'same'
reduced_shape = (14,14)
spacing = 8
n_epochs = 1
alpha = 0.4
lamda_init = 2e-3
decay_rate = 0.01
expdecay_tol = 10
bond_dim = 5
init_func = 'normal'
scale_init_p = 0.5

# %%time

import sys
sys.path.append('../')
from tnad.tnad.optimization import train_SMPO, load_mnist_train_data, data_preprocessing
import tnad.procedures as p

train_data = load_mnist_train_data(train_size=train_size, seed=123456)
data = data_preprocessing(train_data, strides=strides, pool_size=pool_size, padding=padding, reduced_shape=reduced_shape)

opt_procedure = p.automatic_differentiation

P, loss_array = train_SMPO(data, spacing, n_epochs, alpha, opt_procedure, lamda_init, decay_rate, expdecay_tol, bond_dim, init_func, scale_init_p, batch_size, seed=123456)
