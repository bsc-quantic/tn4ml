from typing import Any
import tn4ml.embeddings as embeddings
from tn4ml.models import SpacedMatrixProductOperator
import quimb.tensor as qtn
import math
import numpy as np
import torch
from torch.func import vmap

def squeeze_image(image, k=3, device=torch.device('cpu')):
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
    reshaped_image = torch.zeros(tuple(new_dims), dtype=torch.float32, device=device)

    feature = 0
    for x in range(k):
        for y in range(k):
            if S == 3:
                # 3D image
                for z in range(k):
                    kernel = np.zeros((k,k,k))
                    kernel[x,y,z] = 1.0
                    kernel = np.expand_dims(kernel, axis=-1)
                    kernel = torch.tensor(kernel, dtype=torch.float32, device=device)
                    
                    tensor = torch.zeros(tuple(new_dims[:-1]), dtype=torch.float32, device=device)
                    for i in range(0, image.shape[0], k):
                        for j in range(0, image.shape[1], k):
                            for l in range(0, image.shape[2], k):
                                patch = torch.sum(image[i:i+k, j:j+k, l:l+k, :] * kernel)
                                
                                tensor[i//k, j//k, l//k] = patch
                    reshaped_image[:,:,:,feature] = tensor
                    feature += 1
            else:
                # 2D image
                kernel = np.zeros((k,k))
                kernel[x,y] = 1.0
                #kernel = np.expand_dims(kernel, axis=-1)
                kernel = torch.tensor(kernel, dtype=torch.float32, device=device)
                
                tensor = torch.zeros(tuple(new_dims[:2]), dtype=torch.float32, device=device)
                for i in range(0, image.shape[0], k):
                    for j in range(0, image.shape[1], k):
                            patch = torch.sum(torch.tensordot(image[i:i+k, j:j+k, :], kernel))
                            new_row = i//k
                            new_col = j//k

                            # Create a mask tensor indicating the position to assign the patch - VMAP
                            mask = torch.zeros_like(tensor, dtype=torch.bool, device=device)
                            mask[new_row, new_col] = True
                            tensor = tensor + patch * mask

                #reshaped_image[:,:,feature] = tensor
                # Create a mask tensor indicating the position to assign the patch - VMAP
                mask = torch.zeros_like(reshaped_image, dtype=torch.bool, device=device)
                mask[:,:,feature] = True
                reshaped_image = reshaped_image + tensor.unsqueeze(-1) * mask
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
    for _ in range(S):
        new_dims.append(round(n_mps**(1/S)))
    new_dims.append(n_features)

    reshaped_image = image.reshape(tuple(new_dims))
    averaged_image = torch.mean(reshaped_image, -1)
    averaged_image = (averaged_image - torch.min(averaged_image))/(torch.max(averaged_image) - torch.min(averaged_image))
    averaged_image = torch.unsqueeze(averaged_image, -1)
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
    for _ in range(S):
        new_dims.append(round(n_mps**(1/S)))
    new_dims.append(1) # averaged by feature dimension

    return tuple(new_dims)


class loTeNet(torch.nn.Module):

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
                 output_dim: int,
                 bond_dim: int,
                 kernel: int = 3,
                 virtual_dim: int = 1,
                 phys_dim_input: int = 2,
                 embedding_input: embeddings.Embedding = embeddings.trigonometric(),
                 device: torch.device = torch.device('cpu')):
        
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.kernel = kernel
        self.virtual_dim = virtual_dim
        self.phys_dim_input = phys_dim_input
        self.embedding_input = embedding_input
        self.device = device

        self.S = len(self.input_dim) - 1 # dimensionality of inputs

        new_dim, feature_dim = squeeze_dimensions(self.input_dim, self.kernel)
        N_i = math.prod(new_dim) # number of pixels after squeeze = number of MPSs in 1st layer

        skeletons = []
        params = []
        BNs = [] # Batch Normalization layers
        self.n_layers = 0
        while True:
            # MPS with output index
            layer_i = [SpacedMatrixProductOperator.rand_init(n=feature_dim,\
                                                                spacing=feature_dim,\
                                                                bond_dim = self.bond_dim,\
                                                                phys_dim=(self.phys_dim_input, self.virtual_dim),\
                                                                init_method='random_eye') for _ in range(N_i)]
            # gather params and skeleton of each MPS
            param_i = []; skeleton_i = []
            for n_pixel, layer in enumerate(layer_i):
                param, skeleton = qtn.pack(layer)

                param_dict = {}
                for i, data in param.items():
                    param_dict[f'param_{i}_mps_{n_pixel}_nL{self.n_layers}'] = torch.nn.Parameter(torch.tensor(data, dtype=torch.float32), requires_grad=True)
                param_i.append(torch.nn.ParameterDict(param_dict))
                skeleton_i.append(skeleton)
            skeletons.append(skeleton_i)
            params.append(torch.nn.ParameterList(param_i))
            BNs.append(torch.nn.BatchNorm1d(self.virtual_dim, affine=True, dtype = torch.float32, track_running_stats=True))

            # find shape of output image
            output_shape = (N_i, self.virtual_dim)
            unsqueezed_dim = unsqueezed_dimensions(output_shape, self.S)
            # find new N_i -> number of MPSs in next layer
            new_dim, feature_dim = squeeze_dimensions(unsqueezed_dim, self.kernel)
            N_i = math.prod(new_dim)

            # if we came to last layer where number of MPS = 1 --> finish 
            if N_i == 1:

                self.BNs = BNs
                # last MPS with output_dim = number of classes
                layer = SpacedMatrixProductOperator.rand_init(n=feature_dim,\
                                                                spacing=feature_dim,\
                                                                bond_dim = self.bond_dim,\
                                                                phys_dim=(self.phys_dim_input, self.output_dim),\
                                                                init_method='random_eye')
                param, skeleton = qtn.pack(layer)
                param_dict = {f'param_{i}_mps_{0}_nL{self.n_layers}': torch.nn.Parameter(torch.tensor(data, dtype=torch.float32), requires_grad=True) for i, data in param.items()}
                params.append(torch.nn.ParameterList([torch.nn.ParameterDict(param_dict)]))
                skeletons.append([skeleton])
                
                # save params and skeletons in Module
                self.params = torch.nn.ParameterList(params)
                self.skeletons = skeletons
                self.n_layers += 1
                return
            elif N_i < 1:
                ValueError("Last layer needs to have N_i = 1.")
            
            self.n_layers += 1

    def pass_per_layer(self, input_image, params_i, skeleton_i):

        squeezed_image = squeeze_image(input_image, k=self.kernel, device=self.device)
        squeezed_image = squeezed_image.reshape(math.prod(squeezed_image.shape[:-1]), squeezed_image.shape[-1])

        vector_outputs = []
        for n_pixel, (params, skeleton) in enumerate(zip(params_i, skeleton_i)):
            # embedded input image
            mps_input = embeddings.embed(squeezed_image[n_pixel], self.embedding_input, pytorch=True)

            # return params to quimb
            params = {int(key.split('_')[1]): value for key, value in params.items()}
            
            # unpack model from params and skeleton
            mps_model = qtn.unpack(params, skeleton)
            
            # MPS + MPS_with_output = vector
            output = mps_model.apply(mps_input)^all

            vector_outputs.append(output.data.reshape((self.virtual_dim, )))
        vector_outputs = torch.stack(vector_outputs, dim=0)
        
        return vector_outputs.to(self.device)

    def pass_final_layer(self, input_image, params, skeleton):
        squeezed_image = squeeze_image(input_image, k=self.kernel, device=self.device)
        squeezed_image = squeezed_image.reshape(math.prod(squeezed_image.shape[:-1]), squeezed_image.shape[-1])

        # embed input image
        mps_input = embeddings.embed(squeezed_image[0], self.embedding_input, pytorch=True) # 0 because its the only pixel
        
        # return params to quimb
        params = {int(key.split('_')[1]): value for key, value in params.items()}
        # unpack model from params and skeleton
        mps_model = qtn.unpack(params, skeleton)

        # MPS + MPS_with_output = vector
        output = mps_model.apply(mps_input)^all
        output = output.data.reshape((self.output_dim, ))
        
        return output.to(self.device)

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

        batch_size = x.shape[0]
        for n_layer, (params_i, skeleton_i) in enumerate(zip(self.params, self.skeletons)):
            #print(f'LAYER = {n_layer}')
            if n_layer == self.n_layers:
                x = vmap(self.pass_final_layer, in_dims=(0, None, None))(x, params_i[0], skeleton_i[0])
                return x

            x = vmap(self.pass_per_layer, in_dims=(0, None, None))(x, params_i, skeleton_i)
            
            x_dims = x.shape[1:]

            # reshape
            x = x.reshape(batch_size, x_dims[-1], x_dims[0])

            # batch norm
            x = self.BNs[n_layer](x) # size = (B, n_features, n_mps)

            # reshape back
            x = x.reshape(batch_size, x_dims[0], x_dims[-1])

            # reshape to image-like for next layer
            x = vmap(unsqueeze_image, in_dims=(0, None))(x, self.S)
