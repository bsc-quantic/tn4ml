from typing import Any, Collection
import tn4ml.embeddings as embeddings
from tn4ml.models import SpacedMatrixProductOperator
import quimb.tensor as qtn
import math
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import jax
import optax

def squeeze_3Dimage(image, k=3):
    """
    Squeeze over H,W,D dimensions, but enlarge feature dimension to k**(S), where S=dimensionality of image.

    Parameters
    ----------
    image : np.array
        3D image
    k : int = 3
        Kernel stride and size (if k=3 then kernel has shape (3,3,3) and its moving by stride 3)

    Returns
    -------
    np.array
    """
    S = 3 # 3D image
    H, V, D, C = image.shape
    new_H = H // k
    new_V = V // k
    new_D = D // k
    new_C = C * k**S
        
    reshaped_image = jnp.zeros((new_H, new_V, new_D, new_C))
    
    feature = 0
    for x in range(k):
        for y in range(k):
            for z in range(k):
                kernel = jnp.zeros((k,k,k))
                kernel.at[x,y,z].set(1)
                kernel = jnp.expand_dims(kernel, axis=-1)
                
                tensor = jnp.zeros((new_H, new_V, new_D))
                for i in range(0, H, k):
                    for j in range(0, V, k):
                        for l in range(0, D, k):
                            patch = jnp.sum(image[i:i+k, j:j+k, l:l+k, :] * kernel)
                            
                            tensor.at[i//k, j//k, l//k].set(patch)
                reshaped_image.at[:,:,:,feature].set(tensor)
                feature += 1
    return reshaped_image

def unsqueeze_3Dimage(image, k=3, s=3):
    """
    Unsqueeze over H,W,D dimensions, but average over feature dimension.

    Parameters
    ----------
    image : np.array
        3D image
    k : int = 3
        Kernel stride and size (if k=3 then kernel has shape (3,3,3) and its moving by stride 3)

    Returns
    -------
    np.array
    """
    n_mps, n_features = image.shape
    height = round(n_mps**(1/s))
    width = round(n_mps**(1/s))
    depth = round(n_mps**(1/s))

    reshaped_image = jnp.reshape(image, (height, width, depth, n_features))
    averaged_image = jnp.average(reshaped_image, axis=-1)
    averaged_image = jnp.expand_dims(averaged_image, axis=-1)

    return averaged_image

def squeeze_dimensions(input_dim, k=3):
    """ helper function for initialization """
    if len(input_dim) < 4:
        ValueError('Input is missing one dimension!')
    H, V, D, C = input_dim
    new_H = H // k
    new_V = V // k
    new_D = D // k
    new_C = C * k**3

    return (new_H, new_V, new_D), new_C

def unsqueezed_dimensions(input_dim, S=3):
    """ helper function for initialization """
    n_mps, n_features = input_dim
    height = round(n_mps**(1/S))
    width = round(n_mps**(1/S))
    depth = round(n_mps**(1/S))

    # averaged by last dimension
    return (height, width, depth, 1)

# TODO - maybe use nn.compact instead of setup and call to avoid code duplicates
class loTeNet_3D(nn.Module):

    input_dim: (int, int, int) # dimensionality of input image
    output_dim: int # number of classes
    bond_dim: int
    kernel: int = 3
    virtual_dim: int = 1 # output dimension of MPS_i
    phys_dim_input: int = 2 # physical dimension --> needs to be same as embedding dim
    embedding_input: embeddings.Embedding = embeddings.trigonometric()
    
    # initialization
    def setup(self):

        #self.S = len(self.input_dim) - 1 # dimensionality of inputs
        #self.N_init = math.prod(self.input_dim) # number of pixels in inputs
        
        new_dim, feature_dim = squeeze_dimensions(self.input_dim, self.kernel)
        N_i = math.prod(new_dim) # number of pixels after squeeze = number of MPSs in 1st layer
        
        skeletons = []
        params = []
        self.n_layers = 0
        while True:
            print(f'------ Layer {self.n_layers} ------')
            print(f'N_i = {N_i}, feature_dim = {feature_dim}')
            
            # MPS with output index
            layer_i = [SpacedMatrixProductOperator.rand_distribution(n=feature_dim,\
                                                                spacing=feature_dim,\
                                                                bond_dim = self.bond_dim,\
                                                                phys_dim=(self.phys_dim_input, self.virtual_dim),\
                                                                init_func='normal') for i in range(N_i)]
            
            # gather params and skeleton of each MPS
            param_i = []; skeleton_i = []
            for n_mps, layer in enumerate(layer_i):
                param, skeleton = qtn.pack(layer)

                param_dict = {}
                for i, data in param.items():
                    param_dict[i] = self.param(f'param_{i}_{n_mps}_{self.n_layers}', lambda _: data)
                
                param_i.append(param_dict)
                skeleton_i.append(skeleton)
            
            skeletons.append(skeleton_i)
            params.append(param_i)

            # find shape of output image
            output_shape = (N_i, self.output_dim)
            unsqueezed_dim = unsqueezed_dimensions(output_shape)

            # find new N_i -> number of MPSs in next layer
            new_dim, feature_dim = squeeze_dimensions(unsqueezed_dim, self.kernel)
            N_i = math.prod(new_dim)
            self.n_layers += 1
            
            # if we came to last layer where number of MPS = 1 --> finish 
            if N_i == 1:

                # last MPS with output_dim = number of classes
                layer = SpacedMatrixProductOperator.rand_distribution(n=feature_dim,\
                                                                spacing=feature_dim,\
                                                                bond_dim = self.bond_dim,\
                                                                phys_dim=(self.phys_dim_input, self.output_dim),\
                                                                init_func='normal')
                param, skeleton = qtn.pack(layer)
                param_dict = {i: self.param(f'param_{i}_{n_mps}_{self.n_layers}', lambda _: data)
            for i, data in param.items()}
                params.append(param_dict)
                skeletons.append(skeleton)
                print(f'------ Last {self.n_layers} layer ------')
                print('End initialization')
                
                # save params and skeletons in Module
                self.params = params
                self.skeletons = skeletons
                break
    
    def __call__(self, input_image):
        # FORWARD PASS

        for n_layer, (params_i, skeleton_i) in enumerate(zip(self.params, self.skeletons)):
            print(f'------ Layer {n_layer} ------')

            # if we came to last layer where number of MPS = 1 --> finish     
            if n_layer == self.n_layers:
                squeezed_image = squeeze_3Dimage(input_image, k=self.kernel)
                squeezed_image = squeezed_image.reshape(math.prod(squeezed_image.shape[:-1]), squeezed_image.shape[-1])
                
                # embedded input image
                embedded_vector = embeddings.embed(squeezed_image[0], self.embedding_input)
                mps_input = qtn.MatrixProductState(embedded_vector)
                
                # unpack model from params and skeleton
                mps_model = qtn.unpack(params_i, skeleton_i)

                # MPS + MPS_with_output = vector
                output = mps_model.apply(mps_input)^all
                output = output.data.reshape((self.output_dim, ))
                return output
            
            squeezed_image = squeeze_3Dimage(input_image, k=self.kernel)
            squeezed_image = squeezed_image.reshape(math.prod(squeezed_image.shape[:-1]), squeezed_image.shape[-1])
            
            vector_outputs = []
            for n_mps, (params, skeleton) in enumerate(zip(params_i, skeleton_i)):
                # embedded input image
                embedded_vector = embeddings.embed(squeezed_image[n_mps], self.embedding_input)
                mps_input = qtn.MatrixProductState(embedded_vector)
                
                # unpack model from params and skeleton
                mps_model = qtn.unpack(params, skeleton)
                
                # MPS + MPS_with_output = vector
                output = mps_model.apply(mps_input)^all
                vector_outputs.append(output.data.reshape((self.virtual_dim, )))
            
            # reshape to image-like for next layer
            input_image = unsqueeze_3Dimage(jnp.array(vector_outputs))

# function to create train step
def create_train_step(key, model, optimiser):
  dummy_input = jnp.ones(shape=(32, 32, 32, 1)) # Dummy Input for initialization of MODEL
  params = model.init(key, dummy_input)
  opt_state = optimiser.init(params)

  def loss_fn(params, data, y_true):
    # vmap for batching
    y_pred = jax.vmap(model.apply, in_axes=(None, 0))(params, data)
    loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, y_true).sum(axis=0).mean()
    return loss

  @jax.jit
  def train_step(params, opt_state, data, y_true):
    loss, grads = jax.value_and_grad(loss_fn)(params, data, y_true)

    updates, opt_state = optimiser.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

  return train_step, params, opt_state

def model_train(n_epochs, train_dataset, targets):
    for epoch in range(n_epochs):
        loss_batch = 0
        batch_num = 0
        
        for batch_x, batch_y in zip(list(train_dataset.as_numpy_iterator()), list(targets.as_numpy_iterator())):
            params, opt_state, loss_curr = train_step(params, opt_state, jnp.asarray(batch_x), jnp.asarray(batch_y))
            print(loss_curr)
            loss_batch += loss_curr
            batch_num+=1
        
        print(f'LOSS = {loss_batch}')



        



        