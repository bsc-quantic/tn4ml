import functools

# args
train_size = 1024
batch_size = 32
strides = (2,2)
pool_size = (2,2)
padding = 'same'
reduced_shape = (14,14)
opt_procedure = 'automatic_differentiation'
spacing = 8
n_epochs = 100 # 1
alpha = 0.4
lamda_init = 5e-4
lamda_init_2 = 5e-4
decay_rate = 0.01
expdecay_tol = 10
bond_dim = 5
init_func = 'normal'
scale_init_p = 0.5

insert = 0
save = True
draw = True

extra_pooling = False
if extra_pooling:
    strides = (4,4)
    pool_size = (4,4)
    reduced_shape = (7,7)


optimizer = None # 'adam'
loss_detail=True

# We can activate this to select and test different algorithm depths but preliminar testing shows 1 to be the best.
# alg_depth=1 and alg_depth=2 are extremely close in time for small tensors, though, and for large tensors 2 should have the advantage, so it is set by default.

alg_depth = 1
# alg_depth = int(input("Enter the depth of automatic differentiation: \n 0) use tnopt.optimize \n 1) use tnopt.value_and_grad and convert everytime \n 2) use tnopt.value_and_grad and convert only at the end \n"))
# if alg_depth==0:
#     print('using tnopt.optimize')
# elif alg_depth==1:
#     print('using tnopt.vectorized_value_and_grad and converting each iteration')
# elif alg_depth==2:
#     print('using tnopt.vectorized_value_and_grad and converting only at the end')

# Again, we can activate this section to select and test using jit vs not using jit.
# In this case, not using jit was favourable when doing tests with TNAD and it was set as default. It might become advantageous to use it with very large tensors, but it is not completely clear due to the changing sizes of the operations (adverse for jit)

jit_fn = False
# jit = input("Do you want to use jit? (y/n) \n")
# if jit=='y' or jit=='yes':
#     jit_fn = True
# elif jit=='n' or jit=='no':
#     jit_fn = False

try:
    client
except NameError:
    print('proceeding without Dask client')
    client = None
# Finally, this is how you parallelize with dask! It's set to do it automatically, since it's faster, but to achieve maximum speed we need to optimize the amount of workers based on our pc by instead using the n_workers argument. For example:
# client = Client(n_workers=12) # in the case of Sergi's PC with 12 threads. Otherwise, the following sets an appropriate amount of workers for the cpu available:
# client = Client() # However! Do not run these outside of a notebook. You will get mile-long errors if running this directly on a terminal with python (like "python examples/auto_diff.py")
# the necessary import is: from dask.distributed import Client, LocalCluster

# %%time

from tnad.optimization import train_SMPO, load_mnist_train_data, data_preprocessing
import tnad.procedures as p

train_data = load_mnist_train_data(train_size=train_size, seed=123456)
data = data_preprocessing(train_data, strides=strides, pool_size=pool_size, padding=padding, reduced_shape=reduced_shape)

opt_procedure = functools.partial(p.automatic_differentiation, alg_depth=alg_depth, jit_fn=jit_fn, par_client=client, optimizer=optimizer, loss_detail=loss_detail)

# If we want to run it with dask, we must initialize it as usual and pass the client object as "par_client". Quimb will take care of the rest. See below for an example:
# opt_procedure = functools.partial(p.automatic_differentiation, alg_depth=2, jit_fn=False, par_client=None)

P, loss_array = train_SMPO(data, spacing, n_epochs, alpha, opt_procedure, lamda_init, lamda_init_2, decay_rate, expdecay_tol, bond_dim, init_func, scale_init_p, batch_size, insert, seed=123456)

loss_total = [loss[0] for loss in loss_array]
loss_reg_array = [loss[1] for loss in loss_array]
loss_miss_array = [loss[2] for loss in loss_array]

import quimb as qu

if save:
    qu.save_to_disk(P,'trained_P_global')

from matplotlib import pyplot as plt

# create plot
if plot:
    markersize = 4.
    cut = 80
    end = len(loss_total)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(range(cut, end), loss_total[cut:end], "o", linestyle="--", markersize=markersize, label=r"total loss")
    ax.plot(range(cut, end), loss_reg_array[cut:end], "s", linestyle="--", markersize=markersize, label=r"loss reg")
    ax.plot(range(cut, end), loss_miss_array[cut:end], "d", linestyle="--", markersize=markersize, label=r"loss miss")

    ax.set_xlabel("epoch", fontsize=16)
    ax.set_ylabel("loss", fontsize=16)
    ax.tick_params(labelsize=14)
    plt.subplots_adjust(
        top=0.97,
        bottom=0.14,
        left=0.13,
        right=0.97,
        hspace=0.,
        wspace=0.
    )
    
    plt.legend(loc="best", prop={'size': 17}, handlelength=2)
    plt.savefig("results_folder\loss_tnad_autodif.svg", bbox_inches='tight')
    # plt.savefig(os.path.join(results_folder,"_different_loss_2.png"), dpi=600)