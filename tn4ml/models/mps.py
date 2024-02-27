import itertools
from numbers import Integral
from typing import Collection, Tuple
import numpy as np
import autoray as a
from quimb import *
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState, TensorNetwork, TensorNetwork1DFlat
from .model import Model
from ..util import gramschmidt

class TrainableMatrixProductState(Model, MatrixProductState):
    """A Trainable MatrixProductState class.
    See :class:`quimb.tensor.tensor_1d.MatrixProductState` for explanation of other attributes and methods.
    """

    def __init__(self, arrays, **kwargs):
        # if isinstance(arrays, MatrixProductState):
        #     Model.__init__(self)
        #     return
        Model.__init__(self)
        MatrixProductState.__init__(self, arrays, **kwargs)
    
    def rand_state(L: int, bond_dim: int, phys_dim: int=2, normalize: bool=True, cyclic: bool=False, dtype: str="float64", trans_invar: bool = False, **kwargs):
        rmps = qtn.MPS_rand_state(L, bond_dim=bond_dim,\
                                  phys_dim = phys_dim,\
                                  normalize = normalize,\
                                  cyclic = cyclic,\
                                  dtype = dtype,\
                                 trans_invar = trans_invar,\
                                 **kwargs)
        arrays = rmps.arrays
        return TrainableMatrixProductState(arrays, **kwargs)

        
    def rand_orthogonal(L: int, bond_dim: int, phys_dim: int=2, normalize: bool=True, cyclic: bool=False, dtype: str="float64", init_func: str = "uniform", scale: float = 1.0, seed: int = None, **kwargs):
        
        if cyclic:
            raise NotImplementedError()

        # check for site varying physical dimensions
        if isinstance(phys_dim, Integral):
            phys_dims = itertools.repeat(phys_dim)
        else:
            phys_dims = itertools.cycle(phys_dim)
        # LRP
        arrays=[]
        for i in range(1, L+1):
            p = next(phys_dims)
            if i > L // 2:
                j = (L + 1 - abs(2*i - L - 1)) // 2
            else:
                j = i
            
            chil = min(bond_dim, p**(j-1))
            chir = min(bond_dim, p**j)

            if i > L // 2:
                (chil, chir) = (chir, chil)

            if i == 1:
                (chil, chir) = (chir, 1)

            shape = (chil, chir, p)

            if seed != None:
                A = gramschmidt(randn([shape[0], np.prod(shape[1:])], dtype=dtype, dist=init_func, scale=scale, seed=seed))
            else:
                A = gramschmidt(randn([shape[0], np.prod(shape[1:])], dtype=dtype, dist=init_func, scale=scale))
            
            arrays.append(np.reshape(A, shape))
    
        arrays[0] = np.reshape(arrays[0], (1, min(bond_dim, phys_dim), phys_dim))
        #arrays[L-1] = np.reshape(arrays[L-1], (min(bond_dim, phys_dim), 1, phys_dim))
        
        arrays[0] /= np.sqrt(phys_dim)
            
        rmps = TrainableMatrixProductState(arrays, **kwargs)

        return rmps
    
    def rand_init(L: int, init_func: str = 'random_eye', bond_dim: int = 4, phys_dim: int = 2, cyclic: bool = False, left_canonize: bool = None, **kwargs):
        # Unpack init_method if it is a tuple
        if not isinstance(init_func, str):
            init_str = init_func[0]
            std = init_func[1]
            # if init_str == 'min_random_eye':
            #     init_dim = init_func[2]

            init_func = init_str
        else:
            std = 1e-9

        if init_func not in ['random_eye', 'min_random_eye', 'random_zero']:
            raise ValueError(f"Unknown initialization method: {init_func}")
        
        if cyclic:
            raise NotImplementedError()

        arrays = []
        for i in range(1, L+1):
            shape = (bond_dim, bond_dim, phys_dim)
            if i == 1:
                shape = (1, bond_dim, phys_dim)
            if i == L:
                shape = (bond_dim, 1, phys_dim)
            
            if init_func == 'random_eye':
                #eye_tensor = np.eye(shape[0], shape[1]) # lrp, so lr
                tensor = np.zeros(shape)
                if i == 1:
                    eye_vector = np.eye(1, shape[1])
                    for p in range(shape[-1]):
                        tensor[:, :, p] = eye_vector # 0, :, p
                elif i == L:
                    eye_vector = np.eye(shape[0], 1)
                    for p in range(shape[-1]):
                        tensor[:, :, p] = eye_vector # :, 0, p
                else:
                    eye_tensor = np.eye(shape[0], shape[1])
                    for p in range(shape[-1]):
                        tensor[:, :, p] = eye_tensor # :, 0, p
                
                # Add on a bit of random noise
                tensor += std * np.random.randn(*shape)

            elif init_func == 'random_zero':
                tensor = std * np.random.randn(*shape)

            arrays.append(np.squeeze(tensor))
            
        rmps = TrainableMatrixProductState(arrays, **kwargs)

        # if left_canonize == None:
        #     rmps.normalize()
        # else:
        rmps.left_canonize()
        # rmps.normalize()
        return rmps