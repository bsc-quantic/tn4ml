import math
import numpy as np
import quimb as qu
import quimb.tensor as qtn
import jax
import autoray
import jax.numpy as jnp
import flax.linen as nn
import tn4ml.embeddings as embeddings
from  tn4ml.models import SpacedMatrixProductOperator
from tn4ml.util import *

def calc_num_layers(input_dim, kernel, S):
    new_dim, _ = squeeze_dimensions(input_dim, kernel)
    N_i = math.prod(new_dim)
    
    n_layers = 1

    while True:
        rearanged_dim = rearanged_dimensions((N_i,), S)
        # find new N_i -> number of MPSs in next layer
        new_dim, _ = squeeze_dimensions(rearanged_dim, kernel)
        N_i = math.prod(new_dim)
        if N_i<2:
            break
        else: n_layers += 1
    return n_layers


class MLMPS(nn.Module):

    input_dim: tuple[int, ...] # dimensionality of input image - each dimension is power of self.kernel
    output_dim: int # number of classes
    bond_dim: int
    kernel: int = 3 # for now only one value for all layers
    virtual_dim: int = 1 # output dimension of MPS_i
    phys_dim_input: int = 2 # physical dimension --> needs to be same as embedding dim
    embedding_input: embeddings.Embedding = embeddings.trigonometric()

    def setup(self):
        self.S = len(self.input_dim) - 1 # dimensionality of inputs

        new_dim, feature_dim = squeeze_dimensions(self.input_dim, self.kernel)
        N_i = math.prod(new_dim) # number of pixels after squeeze = number of MPSs in 1st layer

        skeletons = []
        params = []
        BNs = [] # Batch Normalization layers

        # calculate number of layers
        #self.n_layers = calc_num_layers(self.input_dim, self.kernel, self.S)
        #nL = np.log(self.input_dim[0])/np.log(self.kernel)
        self.n_layers = 2

        for i in range(self.n_layers):
            # print(f'N_layer = {i+1}')
            # print(f'N_i, feature_dim = {N_i}, {feature_dim}')
            # MPS with output index
            layer_i = SpacedMatrixProductOperator.rand_init(n=N_i,\
                                                            spacing=N_i,\
                                                            bond_dim = self.bond_dim,\
                                                            phys_dim=(feature_dim, N_i),\
                                                            init_func='random_eye')
            # print(f'#outputs = {len(list(layer_i.lower_inds))}')
            #n_outputs = len(list(layer_i.lower_inds))
            # gather params and skeleton of each MPS
            param, skeleton = qtn.pack(layer_i)
            param_dict = {}
            for j, data in param.items():
                param_dict[j] = self.param(f'param_{j}_nL{i}', lambda _: data)
            params.append(param_dict)
            skeletons.append(skeleton)

            # find shape of output image
            output_shape = (N_i, )
            rearanged_dim = rearanged_dimensions(output_shape, self.S)
            # print(f"Rearanged dim H' x W' = {rearanged_dim}")
            # find new N_i -> number of MPSs in next layer
            new_dim, feature_dim = squeeze_dimensions(rearanged_dim, self.kernel)
            N_i = math.prod(new_dim)
        
        # print('OUTPUT')
        # print(f'n_layer = {self.n_layers+1}')
        #print(f'Number of tensors in last layer = {flatten_dim}')
        layer_i = SpacedMatrixProductOperator.rand_init(n=N_i,\
                                                        spacing=N_i,\
                                                        bond_dim = self.bond_dim,\
                                                        phys_dim=(feature_dim, self.output_dim),\
                                                        init_func='random_eye')
        # print(f'#outputs = {len(list(layer_i.lower_inds))}')
        param, skeleton = qtn.pack(layer_i)
        param_dict = {}
        for j, data in param.items():
            param_dict[j] = self.param(f'param_{j}_nL{self.n_layers+1}', lambda _: data)
        params.append(param_dict)
        skeletons.append(skeleton)

        self.params = params
        self.skeletons = skeletons

    def pass_first_layer(self, input_image, params, skeleton):
        squeezed_image = squeeze_image(input_image, k=self.kernel)
        squeezed_image = squeezed_image.reshape(math.prod(squeezed_image.shape[:-1]), squeezed_image.shape[-1])

        # embedded input image
        N_i, feature_dim = squeezed_image.shape
        mps_input = embeddings.embed(squeezed_image, phi_multidim=self.embedding_input)
        mps_input.normalize()
        
        # unpack model from params and skeleton
        mps_model = qtn.unpack(params, skeleton)
        mps_model.normalize()

        # MPS + MPS_with_output = vector
        output = mps_model.apply(mps_input)

        # Iteratively contract the result with each subsequent tensor
        with autoray.backend_like("jax"), qtn.contract_backend("jax"):
            result = output[0]
            for i in range(1, len(output.tensors)):
                result = result @ output[i]
                new_inds = [ind for ind, size in zip(result.inds, result.shape) if size > 1]
                # Corresponding sizes for the new shape
                new_shape = [size for size in result.shape if size > 1]
                result.modify(data=result.data.reshape(new_shape), inds=new_inds)

            result.drop_tags(result.tags)
            result.add_tag(['I0'])
            result = result.data.reshape((N_i, ))
        return result
    
    def pass_per_layer(self, input_image, params, skeleton):
        squeezed_image = squeeze_image(input_image, k=self.kernel)
        squeezed_image = squeezed_image.reshape(math.prod(squeezed_image.shape[:-1]), squeezed_image.shape[-1])

        # embedded input image
        N_i, feature_dim = squeezed_image.shape
        #mps_input = embeddings.embed(squeezed_image, self.embedding_input)

        arrays = []
        for pixel in squeezed_image:
            arrays.append(pixel.reshape((1,1,feature_dim)))
        
        for i in [0, -1]:
            arrays[i] = arrays[i].reshape((1, feature_dim))
        mps_input = qtn.MatrixProductState(arrays)
        mps_input.normalize()
        
        # unpack model from params and skeleton
        mps_model = qtn.unpack(params, skeleton)
        mps_model.normalize()

        # MPS + MPS_with_output = vector
        output = mps_model.apply(mps_input)
        

        # Iteratively contract the result with each subsequent tensor
        with autoray.backend_like("jax"), qtn.contract_backend("jax"):
            result = output[0]
            for i in range(1, len(output.tensors)):
                result = result @ output[i]
                
                new_inds = [ind for ind, size in zip(result.inds, result.shape) if size > 1]
                # Corresponding sizes for the new shape
                new_shape = [size for size in result.shape if size > 1]
                result.modify(data=result.data.reshape(new_shape), inds=new_inds)

            result.drop_tags(result.tags)
            result.add_tag(['I0'])
            result = result.data.reshape((N_i, ))
        return result
    
    def pass_final_layer(self, input_image, params, skeleton):
        squeezed_image = squeeze_image(input_image, k=self.kernel)
        squeezed_image = squeezed_image.reshape(math.prod(squeezed_image.shape[:-1]), squeezed_image.shape[-1])

        # embedded input image
        # flatten_image = input_image.flatten()
        N_i, feature_dim = squeezed_image.shape
        #mps_input = embeddings.embed(squeezed_image, self.embedding_input)
        arrays = []
        for pixel in squeezed_image:
            arrays.append(pixel.reshape((1,1,feature_dim)))
        
        for i in [0, -1]:
            arrays[i] = arrays[i].reshape((1, feature_dim))
        
        mps_input = qtn.MatrixProductState(arrays)
        mps_input.normalize()
        
        # unpack model from params and skeleton
        mps_model = qtn.unpack(params, skeleton)
        mps_model.normalize()

        # MPS + MPS_with_output = vector
        output = mps_model.apply(mps_input)
        
        with autoray.backend_like("jax"), qtn.contract_backend("jax"):
            result = output[0]
            # Iteratively contract the result with each subsequent tensor
            for i in range(1, len(output.tensors)):
                result = result @ output[i]
                new_inds = [ind for ind, size in zip(result.inds, result.shape) if size > 1]
                # Corresponding sizes for the new shape
                new_shape = [size for size in result.shape if size > 1]
                result.modify(data=result.data.reshape(new_shape), inds=new_inds)

            result.drop_tags(result.tags)
            result.add_tag(['I0'])
            result = result.data.reshape((self.output_dim, ))
        return result


    def __call__(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : np.array
            2D or 3D image with additional Channel dimension (H,W,C) or (H,W,D,C)
        S : int = 3
            Dimensionality of input image. S = 3 --> 3D image

        Returns
        -------
        np.array
        
        """

        for n_layer, (params_i, skeleton_i) in enumerate(zip(self.params, self.skeletons)):
            if n_layer == 0:
                x = self.pass_first_layer(x, params_i, skeleton_i)
            elif n_layer == self.n_layers:
                x = self.pass_final_layer(x, params_i, skeleton_i)
                return x
            else:
                x = self.pass_per_layer(x, params_i, skeleton_i)

            x = rearange_image(x, self.S)


