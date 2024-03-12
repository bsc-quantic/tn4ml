import torch
import math
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from torch.func import vmap
import tn4ml.embeddings as embeddings
from  tn4ml.models import SpacedMatrixProductOperator
from tn4ml.util import squeeze_dimensions, rearanged_dimensions, squeeze_image, rearange_image, unsqueeze_image_pooling, unsqueezed_dimensions


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


class MLSMPO(torch.nn.Module):

    # input_dim: dimensionality of input image - each dimension is power of self.kernel
    # output_dim: number of classes
    # bond_dim: bond dimension
    # kernel: kernel value - both padding and stride, for now only one value for all layers
    # virtual_dim: output dimension of MPS_i
    # phys_dim_input: physical dimension --> needs to be same as embedding dim
    # embedding_input: embedding of input images to MPS

    # initialization
    def __init__(self,
                 input_dim: tuple[int, ...],
                 spacing: int,
                 output_dim: int,
                 bond_dim: int,
                 virtual_dim: int = 1,
                 kernels: tuple[int, ...] = [2, 2],
                 phys_dim_input: int = 2,
                 embedding_input: embeddings.Embedding = embeddings.trigonometric(),
                 device: torch.device = torch.device('cpu')):
        
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.virtual_dim = virtual_dim
        self.kernels = kernels
        self.phys_dim_input = phys_dim_input
        self.embedding_input = embedding_input
        self.device = device
        self.spacing = spacing

        self.S = len(self.input_dim) - 1 # dimensionality of inputs

        new_dim, feature_dim = squeeze_dimensions(self.input_dim, self.kernels[0])
        N_i = math.prod(new_dim) # number of pixels after squeeze = number of MPSs in 1st layer

        skeletons = []
        params = []
        BNs = [] # Batch Normalization layers
        
        # calculate number of layers
        #self.n_layers = calc_num_layers(self.input_dim, self.kernel, self.S)
        #nL = np.log(self.input_dim[0])/np.log(self.kernel)
        self.n_layers = 2

        for i in range(self.n_layers-1):
            print(f'N_layer = {i}')
            print(f'N_i, feature_dim = {N_i}, {feature_dim}')
            # MPS with output index
            layer_i = SpacedMatrixProductOperator.rand_init(n=N_i,\
                                                            spacing=self.spacing,\
                                                            bond_dim = self.bond_dim,\
                                                            phys_dim=(feature_dim, self.virtual_dim),\
                                                            init_func='random_eye')
            print(f'#outputs = {len(list(layer_i.lower_inds))}')
            
            n_outputs = len(list(layer_i.lower_inds))
            # gather params and skeleton of each MPS
            param, skeleton = qtn.pack(layer_i)

            param_dict = {}
            for j, data in param.items():
                param_dict[f'param_{j}_nL{i}'] = torch.nn.Parameter(torch.tensor(data, dtype=torch.float64), requires_grad=True)
            params.append(torch.nn.ParameterDict(param_dict))
            skeletons.append(skeleton)

            # add BatchNormalization
            BNs.append(torch.nn.BatchNorm1d(self.virtual_dim**n_outputs, affine=True, dtype = torch.float64, device=self.device, track_running_stats=True))

            # find shape of output image
            output_shape = (self.virtual_dim**n_outputs,)
            print(f'Output vector shape = {output_shape}')
            rearanged_dim = rearanged_dimensions(output_shape, S=2)
            print(f"Rearanged dim H' x W' = {rearanged_dim}")
            # find new N_i -> number of MPSs in next layer
            new_dim, feature_dim = squeeze_dimensions(rearanged_dim, self.kernels[1])
            N_i = math.prod(new_dim)
        
        print('OUTPUT')
        print(f'n_layer = {self.n_layers}')
        print(f'N_i, feature_dim = {N_i}, {feature_dim}')
        # NEW
        print(f'Number of tensors in last layer = {N_i}')
        layer_i = SpacedMatrixProductOperator.rand_orthogonal(n=N_i,\
                                                        spacing=N_i,\
                                                        bond_dim = self.bond_dim,\
                                                        phys_dim=(feature_dim, self.output_dim),\
                                                        init_func='normal')
        print(f'#outputs = {len(list(layer_i.lower_inds))}')
        param, skeleton = qtn.pack(layer_i)
        param_dict = {}
        for j, data in param.items():
            param_dict[f'param_{j}_nL{self.n_layers}'] = torch.nn.Parameter(torch.tensor(data, dtype=torch.float64), requires_grad=True)
        params.append(torch.nn.ParameterDict(param_dict))
        skeletons.append(skeleton)
        
        # save params and skeletons in Module
        self.BNs = BNs
        self.params = torch.nn.ParameterList(params)
        self.skeletons = skeletons
        return
    
    def pass_per_layer(self, input_image, n_layer, params, skeleton):

        squeezed_image = squeeze_image(input_image, k=self.kernels[n_layer], device=self.device)
        squeezed_image = squeezed_image.reshape(math.prod(squeezed_image.shape[:-1]), squeezed_image.shape[-1])

        # embedded input image
        N_i, feature_dim = squeezed_image.shape
        embedding = embeddings.whatever_encoding(dim=feature_dim)
        mps_input = embeddings.embed(squeezed_image, embedding)
        mps_input.normalize()

        # return params to quimb
        params_quimb = {int(key.split('_')[1]): value for key, value in params.items()}

        # unpack model from params and skeleton
        mps_model = qtn.unpack(params_quimb, skeleton)
        mps_model.normalize()
        
        n_outputs = len(list(mps_model.lower_inds))

        # MPS + MPS_with_output = vector
        output = mps_model.apply(mps_input)
        output.normalize()

        # # Iteratively contract the result with each subsequent tensor
        result = output[0]
        # Iteratively contract the result with each subsequent tensor
        for i in range(1, len(output.tensors)):
            result = result.contract(output[i])
            new_inds = [ind for ind, size in zip(result.inds, result.shape) if size > 1]
            # Corresponding sizes for the new shape
            new_shape = [size for size in result.shape if size > 1]
            result = qtn.Tensor(result.data.reshape(new_shape), inds=new_inds, tags=result.tags)

        result.drop_tags(result.tags)
        result.add_tag(['I0'])
        result = result.data.reshape((self.virtual_dim**n_outputs,))
        return result.to(device=self.device)
    
    def pass_final_layer(self, input_image, params, skeleton):

        squeezed_image = squeeze_image(input_image, k=self.kernels[-1], device=self.device)
        squeezed_image = squeezed_image.reshape(math.prod(squeezed_image.shape[:-1]), squeezed_image.shape[-1])

        # embedded input image
        N_i, feature_dim = squeezed_image.shape
        embedding = embeddings.whatever_encoding(dim=feature_dim)
        mps_input = embeddings.embed(squeezed_image, embedding)
        mps_input.normalize()
        
        # return params to quimb
        params_quimb = dict()
        for key, value in params.items():
            params_quimb[int(key.split('_')[1])] = value
        # unpack model from params and skeleton
        mps_model = qtn.unpack(params_quimb, skeleton)
        mps_model.normalize()
        
        # MPS + MPS_with_output = vector
        output = mps_model.apply(mps_input)^all
        #output.normalize()
        return output.data.reshape((self.output_dim,)).to(device=self.device)
    
    def forward(self, x):
        """
        Forward pass - per batch

        Parameters
        ----------
        image : torch.tensor
            2D or 3D image with additional Channel dimension (B,H,W,C) or (B,H,W,D,C)
            where B = batch_size

        Returns
        -------
        torch.tensor
        
        """

        #batch_size = x.shape[0]
        for n_layer, (params_i, skeleton_i) in enumerate(zip(self.params, self.skeletons)):
            if n_layer == self.n_layers - 1:
                x = vmap(self.pass_final_layer, in_dims=(0, None, None))(x, params_i, skeleton_i)
                return x
            
            x = vmap(self.pass_per_layer, in_dims=(0, None, None, None))(x, n_layer, params_i, skeleton_i)
            
            #x_dims = x.shape[1:]

            # reshape
            #x = x.reshape(batch_size, x_dims[-1], x_dims[0]).to(self.device)

            # batch norm
            x = self.BNs[n_layer](x) # size = (B, n_features, n_mps)
            # reshape back
            #x = x.reshape(batch_size, x_dims[0], x_dims[-1]).to(self.device)

            # reshape to image-like for next layer
            #x = torch.unsqueeze(x, -1)
            x = vmap(rearange_image, in_dims=(0, None, None))(x, self.S, self.device)
            #print(x.shape)
