import functools

# args
train_size = 1024
batch_size = 32
strides = (2,2)
pool_size = (2,2)
padding = 'same'
reduced_shape = (14,14)
opt_procedure = 'global_update_costfuncnorm'
spacing = 8
n_epochs = 100
alpha = 0.4
lamda_init = 5e-4
lamda_init_2 = 5e-4
decay_rate = 0.01
expdecay_tol = 20
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

# %%time

from tnad.optimization import train_SMPO, load_mnist_train_data, data_preprocessing
import tnad.procedures as p

train_data = load_mnist_train_data(train_size=train_size, seed=123456)
data = data_preprocessing(train_data, strides=strides, pool_size=pool_size, padding=padding, reduced_shape=reduced_shape)

opt_procedure = p.global_update_costfuncnorm

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
    plt.savefig("results_folder\loss_tnad_global.svg", bbox_inches='tight')
    # plt.savefig(os.path.join(results_folder,"_different_loss_2.png"), dpi=600)


# example with extra reduction (so: (7,7) size)
# [9.227739334106445, 1.8717690706253052, 1.0751664638519287, 1.193156361579895, 1.665894627571106, 0.6311149597167969, 0.8400715589523315, 1.116007924079895, 0.7961055040359497, 0.5186892151832581, 0.7392604351043701, 0.8423198461532593, 1.11724054813385, 0.7088192701339722, 0.5977280139923096, 0.5282566547393799, 2.4948158264160156, 1.4086428880691528, 0.9442130327224731, 1.0813006162643433, 0.7172842621803284, 1.1994593143463135, 0.7225455045700073, 0.5235581994056702, 1.0108939409255981, 0.7419005036354065, 0.5570693016052246, 0.6137683391571045, 0.7392929196357727, 0.7979183793067932, 0.5951648950576782, 0.3461158573627472]