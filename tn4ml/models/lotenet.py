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

def squeeze_image(image, k=3):
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
    S = len(image.shape) - 1 # dimensionality of image
    if S < 2:
        ValueError('Image should be 2D or 3D!')

    new_dims = []
    for dim in list(image.shape)[:-1]:
        new_dims.append(dim // k)
    new_dims.append(list(image.shape)[-1] * k**S)
        
    reshaped_image = jnp.zeros(tuple(new_dims))
    
    feature = 0
    for x in range(k):
        for y in range(k):
            if S == 3:
                # 3D image
                for z in range(k):
                    kernel = jnp.zeros((k,k,k))
                    kernel.at[x,y,z].set(1)
                    kernel = jnp.expand_dims(kernel, axis=-1)
                    
                    tensor = jnp.zeros(tuple(new_dims[:-1]))
                    for i in range(0, image.shape[0], k):
                        for j in range(0, image.shape[1], k):
                            for l in range(0, image.shape[2], k):
                                patch = jnp.sum(image[i:i+k, j:j+k, l:l+k, :] * kernel)
                                
                                tensor.at[i//k, j//k, l//k].set(patch)
                    reshaped_image.at[:,:,:,feature].set(tensor)
                    feature += 1
            else:
                # 2D image
                kernel = jnp.zeros((k,k))
                kernel.at[x,y].set(1)
                kernel = jnp.expand_dims(kernel, axis=-1)
                
                tensor = jnp.zeros(tuple(new_dims[:2]))
                for i in range(0, image.shape[0], k):
                    for j in range(0, image.shape[1], k):
                            patch = jnp.sum(image[i:i+k, j:j+k, :] * kernel)
                            
                            tensor.at[i//k, j//k].set(patch)
                reshaped_image.at[:,:,feature].set(tensor)
                feature += 1
    return reshaped_image

def unsqueeze_image(image, S=3):
    """
    Unsqueeze over H,W,(D) dimensions, but average over feature dimension.

    Parameters
    ----------
    image : np.array
        2D or 3D image
    S : int = 3
        Dimensionality of input image. S = 3 --> 3D image

    Returns
    -------
    np.array
    """
    n_mps, n_features = image.shape
    new_dims = []
    for dim in range(S):
        new_dims.append(round(n_mps**(1/S)))
    new_dims.append(n_features)

    reshaped_image = jnp.reshape(image, tuple(new_dims))
    averaged_image = jnp.average(reshaped_image, axis=-1)
    averaged_image = jnp.expand_dims(averaged_image, axis=-1)

    return averaged_image

def squeeze_dimensions(input_dims, k=3):
    """ helper function for initialization """
    if len(input_dims) < 3:
        ValueError('Input data must have at least 2 dimensions and one channel')
    S = len(input_dims) - 1

    new_dims = []
    for dim in input_dims[:-1]:
        new_dims.append(dim // k)
    
    feature_dim = input_dims[-1] * k**S # C = input_dims[-1]

    return tuple(new_dims), feature_dim

def unsqueezed_dimensions(input_dims, S=3):
    """ helper function for initialization """
    n_mps, _ = input_dims

    new_dims = []
    for i in range(S):
        new_dims.append(round(n_mps**(1/S)))
    new_dims.append(1) # averaged by feature dimension

    return tuple(new_dims)

# TODO - maybe use nn.compact instead of setup and call to avoid code duplicates
class loTeNet(nn.Module):

    input_dim: tuple[int, ...] # dimensionality of input image
    output_dim: int # number of classes
    bond_dim: int
    kernel: int = 3
    virtual_dim: int = 1 # output dimension of MPS_i
    phys_dim_input: int = 2 # physical dimension --> needs to be same as embedding dim
    embedding_input: embeddings.Embedding = embeddings.trigonometric()
    
    # initialization
    def setup(self):

        self.S = len(self.input_dim) - 1 # dimensionality of inputs
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
            for n_pixel, layer in enumerate(layer_i):
                param, skeleton = qtn.pack(layer)

                param_dict = {}
                for i, data in param.items():
                    param_dict[i] = self.param(f'param_{i}_{n_pixel}_{self.n_layers}', lambda _: data)
                
                param_i.append(param_dict)
                skeleton_i.append(skeleton)
            
            skeletons.append(skeleton_i)
            params.append(param_i)

            # find shape of output image
            output_shape = (N_i, self.output_dim)
            unsqueezed_dim = unsqueezed_dimensions(output_shape, self.S)

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
                param_dict = {i: self.param(f'param_{i}_{1}_{self.n_layers}', lambda _: data)
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
        """
        Forward pass.

        Parameters
        ----------
        image : np.array
            2D or 3D image with additional Channel dimension (H,W,C) or (H,W,D,C)
        S : int = 3
            Dimensionality of input image. S = 3 --> 3D image

        Returns
        -------
        np.array
        
        """
        # FORWARD PASS

        for n_layer, (params_i, skeleton_i) in enumerate(zip(self.params, self.skeletons)):
            print(f'------ Layer {n_layer} ------')

            # if we came to last layer where number of MPS = 1 --> finish     
            if n_layer == self.n_layers:
                squeezed_image = squeeze_image(input_image, k=self.kernel)
                squeezed_image = squeezed_image.reshape(math.prod(squeezed_image.shape[:-1]), squeezed_image.shape[-1])
                
                # embed input image
                embedded_vector = embeddings.embed(squeezed_image[0], self.embedding_input) # 0 because its the only pixel
                mps_input = qtn.MatrixProductState(embedded_vector)
                
                # unpack model from params and skeleton
                mps_model = qtn.unpack(params_i, skeleton_i)

                # MPS + MPS_with_output = vector
                output = mps_model.apply(mps_input)^all
                output = output.data.reshape((self.output_dim, ))
                return output
            
            squeezed_image = squeeze_image(input_image, k=self.kernel)
            squeezed_image = squeezed_image.reshape(math.prod(squeezed_image.shape[:-1]), squeezed_image.shape[-1])
            
            vector_outputs = []
            for n_pixel, (params, skeleton) in enumerate(zip(params_i, skeleton_i)):
                # embedded input image
                embedded_vector = embeddings.embed(squeezed_image[n_pixel], self.embedding_input)
                mps_input = qtn.MatrixProductState(embedded_vector)
                
                # unpack model from params and skeleton
                mps_model = qtn.unpack(params, skeleton)
                
                # MPS + MPS_with_output = vector
                output = mps_model.apply(mps_input)^all
                vector_outputs.append(output.data.reshape((self.virtual_dim, )))
            
            # reshape to image-like for next layer
            input_image = unsqueeze_image(jnp.array(vector_outputs), self.S)


        



        