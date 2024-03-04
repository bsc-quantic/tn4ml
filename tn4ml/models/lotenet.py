from typing import Any
import tn4ml.embeddings as embeddings
from tn4ml.models import SpacedMatrixProductOperator
import quimb.tensor as qtn
import math
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from tn4ml.util import *

# TODO - maybe use nn.compact instead of setup and call to avoid code duplicates
class loTeNet(nn.Module):

    input_dim: tuple[int, ...] # dimensionality of input image - each dimension is power of self.kernel
    output_dim: int # number of classes
    bond_dim: int
    kernel: int = 3 # for now only one value for all layers
    virtual_dim: int = 1 # output dimension of MPS_i
    phys_dim_input: int = 2 # physical dimension --> needs to be same as embedding dim
    embedding_input: embeddings.Embedding = embeddings.trigonometric()
    
    # initialization
    def setup(self):

        self.S = len(self.input_dim) - 1 # dimensionality of inputs

        new_dim, feature_dim = squeeze_dimensions(self.input_dim, self.kernel)
        N_i = math.prod(new_dim) # number of pixels after squeeze = number of MPSs in 1st layer
        
        skeletons = []
        params = []
        self.n_layers = 0
        while True:
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
                #print(f'------ Last {self.n_layers} layer ------')
                #print('End initialization')
                
                # save params and skeletons in Module
                self.params = params
                self.skeletons = skeletons
                break
            elif N_i < 1:
                ValueError("Last layer needs to have N_i = 1.")
    
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
            input_image = unsqueeze_image_pooling(jnp.array(vector_outputs), self.S)

        



        