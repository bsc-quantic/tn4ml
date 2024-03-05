import itertools
from numbers import Integral
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
        """Initialize randomly tensors - taken from quimb.tensor.MPS_rand_state().

        Parameters
        ----------
        L: int
            Number of tensors.
        bond_dim : int
            Dimension of virtual indices between tensors. *Default = 2*.
        phys_dim :  int
            Dimension of physical indices for individual tensor - *up* and *down*.
        normalize : bool
            Flag to normlize tensor network.
        cyclic : bool
            Flag for indicating if SpacedMatrixProductOperator is cyclic. *Default=False*.
        trans_invar : bool
            Type of random number for generating arrays data. *Default='uniform'*.

        Returns
        -------
        :class:`tn4ml.models.mps.TrainableMatrixProductState`
        """

        rmps = qtn.MPS_rand_state(L, bond_dim=bond_dim,\
                                  phys_dim = phys_dim,\
                                  normalize = normalize,\
                                  cyclic = cyclic,\
                                  dtype = dtype,\
                                 trans_invar = trans_invar,\
                                 **kwargs)
        arrays = rmps.arrays
        return TrainableMatrixProductState(arrays, **kwargs)

        
    def rand_orthogonal(L: int, bond_dim: int = 2, phys_dim: int = 2, normalize: bool=True, cyclic: bool=False, dtype: str="float64", init_func: str = "uniform", scale: float = 1.0, seed: int = None, **kwargs):
        """Initialize tensors with Gram-Schmidt ortogonalization procedure ensuring normalized state.
        Currently this function is only supported for `cyclic=False`.

        Parameters
        ----------
        L: int
            Number of tensors.
        bond_dim : int
            Dimension of virtual indices between tensors. *Default = 2*.
        phys_dim :  int
            Dimension of physical indices for individual tensor - *up* and *down*.
        normalize : bool
            Flag to normlize tensor network.
        cyclic : bool
            Flag for indicating if SpacedMatrixProductOperator is cyclic. *Default=False*.
        init_func : str
            Type of random number for generating arrays data. *Default='uniform'*.
        scale : float
            The width of the distribution (standard deviation if `init_func='normal'`).
        seed : int, or `None`
            Seed for generating random number.
        dtype : str 
            Type of random number for quimb.randn()

        Returns
        -------
        :class:`tn4ml.models.mps.TrainableMatrixProductState`
        """

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
        
        if normalize:
            rmps.normalize()
        return rmps