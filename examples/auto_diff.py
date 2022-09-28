import functools

# args
train_size = 1024
batch_size = 32
strides = (4,4) # (2,2)
pool_size = (4,4) # (2,2)
padding = 'same'
reduced_shape = (7,7) # (14,14)
spacing = 8
n_epochs = 2 # 1
alpha = 0.4
lamda_init = 2e-3
decay_rate = 0.01
expdecay_tol = 10
bond_dim = 5
init_func = 'normal'
scale_init_p = 0.5

# We can activate this to select and test different algorithm depths but preliminar testing shows 1 to be the best.
# alg_depth=1 and alg_depth=2 are extremely close in time for small tensors, though, and for large tensors 2 should have the advantage, so it is set by default.

alg_depth = 2
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

# %%time

from tnad.optimization import train_SMPO, load_mnist_train_data, data_preprocessing
import tnad.procedures as p

train_data = load_mnist_train_data(train_size=train_size, seed=123456)
data = data_preprocessing(train_data, strides=strides, pool_size=pool_size, padding=padding, reduced_shape=reduced_shape)

opt_procedure = functools.partial(p.automatic_differentiation, alg_depth=alg_depth, jit_fn=jit_fn)

# If we want to run it with dask, we must initialize it as usual and pass the client object as "par_client". Quimb will take care of the rest. See below for an example:
# opt_procedure = functools.partial(p.automatic_differentiation, alg_depth=2, jit_fn=False, par_client=None)

P, loss_array = train_SMPO(data, spacing, n_epochs, alpha, opt_procedure, lamda_init, decay_rate, expdecay_tol, bond_dim, init_func, scale_init_p, batch_size, seed=123456)

# example with extra reduction (so: (7,7) size)
# [9.227739334106445, 1.8717690706253052, 1.0751664638519287, 1.193156361579895, 1.665894627571106, 0.6311149597167969, 0.8400715589523315, 1.116007924079895, 0.7961055040359497, 0.5186892151832581, 0.7392604351043701, 0.8423198461532593, 1.11724054813385, 0.7088192701339722, 0.5977280139923096, 0.5282566547393799, 2.4948158264160156, 1.4086428880691528, 0.9442130327224731, 1.0813006162643433, 0.7172842621803284, 1.1994593143463135, 0.7225455045700073, 0.5235581994056702, 1.0108939409255981, 0.7419005036354065, 0.5570693016052246, 0.6137683391571045, 0.7392929196357727, 0.7979183793067932, 0.5951648950576782, 0.3461158573627472]