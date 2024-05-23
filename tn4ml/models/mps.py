from typing import Any
import numpy as np
import autoray as a
import math

from quimb import *
import quimb.tensor as qtn

from jax.nn.initializers import Initializer
import jax.numpy as jnp

from .model import Model

class MatrixProductState(Model, qtn.MatrixProductState):
    """A Trainable MatrixProductState class.
    See :class:`quimb.tensor.tensor_1d.MatrixProductState` for explanation of other attributes and methods.
    """

    def __init__(self, arrays, **kwargs):
        """Initializes :class:`tn4ml.models.mps.MatrixProductState`.
        
        Parameters
        ----------
        arrays : list
            List of arrays to be used as tensors.
        kwargs : dict
            Additional arguments.
        """
        Model.__init__(self)
        qtn.MatrixProductState.__init__(self, arrays, **kwargs)
    
    # def copy(self):
    #     """Copies the model.
        
    #     Returns
    #     -------
    #     Model of the same type.
    #     """

    #     model = type(self)(self.arrays)
    #     for key in self.__dict__.keys():
    #         model.__dict__[key] = self.__dict__[key]
    #     return model
    
def trainable_wrapper(mps: qtn.MatrixProductState, **kwargs) -> MatrixProductState:
    """ Creates a wrapper around qtn.MatrixProductState so it can be trainable.

    Parameters
    ----------
    mps : :class:`quimb.tensor.MatrixProductState`
        Matrix Product State to be trained.

    Returns
    -------
    :class:`tn4ml.models.mps.MatrixProductState`
    """
    tensors = mps.arrays
    return MatrixProductState(tensors, **kwargs)
    
def generate_shape(method: str,
                    L: int,
                    bond_dim: int = 2,
                    phys_dim: int = 2,
                    cyclic: bool = False,
                    position: int = None,
                    class_dim: int = None
                    ) -> tuple:
    """Returns a shape of tensor .

    Parameters
    ----------
    method : str
        Method on how to create shapes of tensors.  
        'even' = exact dimensions as given by parameters, anything else = truncated dimensions.
    L : int
        Number of tensors.
    bond_dim : int
        Dimension of virtual indices between tensors. *Default = 4*.
    phys_dim :  int
        Dimension of physical index for individual tensor.
    cyclic : bool
        Flag for indicating if MatrixProductState this tensor is part of is cyclic. *Default=False*.
    position : int
        Position of tensor in MatrixProductState.
    Returns
    -------
        tuple
    """
    
    if class_dim is None:
        if method == 'even':
            shape = (bond_dim, bond_dim, phys_dim)
            if position == 1:
                shape = (1, bond_dim, phys_dim)
            if position == L:
                shape = (bond_dim, 1, phys_dim)
        else:
            assert not cyclic
            if position > L // 2:
                j = (L + 1 - abs(2*position - L - 1)) // 2
            else:
                j = position
            
            chir = min(bond_dim, phys_dim**j)
            chil = min(bond_dim, phys_dim**(j-1))

            if position > L // 2:
                (chil, chir) = (chir, chil)

            if position == 1:
                (chil, chir) = (chir, 1)

            shape = (chil, chir, phys_dim)
    else:
        if method == 'even':
            shape = (bond_dim, bond_dim, phys_dim, class_dim)
            if position == 1:
                shape = (1, bond_dim, phys_dim, class_dim)
            if position == L:
                shape = (bond_dim, 1, phys_dim, class_dim)
        else:
            assert not cyclic
            if position > L // 2:
                j = (L + 1 - abs(2*position - L - 1)) // 2
            else:
                j = position

            chir = min(bond_dim, phys_dim ** (j) * class_dim ** j)
            chil = min(bond_dim, phys_dim ** (j-1) * class_dim ** (j-1))

            if position > L // 2:
                (chil, chir) = (chir, chil)

            if position == 1:
                shape = (chir, phys_dim, class_dim)
            elif position == L:
                shape = (chil, phys_dim, class_dim)
            else:
                shape = (chil, chir, phys_dim, class_dim)
    return shape

def MPS_initialize(L: int,
            initializer: Initializer,
            key: Any,
            dtype: Any = jnp.float_,
            shape_method: str = 'even',
            bond_dim: int = 4,
            phys_dim: int = 2,
            cyclic: bool = False,
            compress: bool = False,
            insert: int = None,
            canonical_center: int = None,
            index_class: int = None,
            class_dim: int = None,
            **kwargs):
    
    """Generates :class:`tn4ml.models.mps.MatrixProductState`.

    Parameters
    ----------
    L : int
        Number of tensors.
    initializer : :class:`jax.nn.initializers.Initializer``
        Type of tensor initialization function.
    key : Array
        Argument key is a PRNG key (e.g. from `jax.random.key()`), used to generate random numbers to initialize the array.
    dtype : Any
        Type of tensor data (from `jax.numpy.float_`)
    shape_method : str
        Method to generate shapes for tensors.
    bond_dim : int
        Dimension of virtual indices between tensors. *Default = 4*.
    phys_dim :  int
        Dimension of physical index for individual tensor.
    cyclic : bool
        Flag for indicating if MatrixProductState is cyclic. *Default=False*.
    compress : bool
        Flag to truncate bond dimensions.
    insert : int
        Index of tensor divided by norm. When `None` the norm division is distributed across all tensors
    canonical_center : int
        If not `None` then create canonical form around canonical center index.

    Returns
    -------
    :class:`tn4ml.models.mps.MatrixProductState`
    """

    if cyclic and shape_method != 'even':
        raise NotImplementedError("Change shape_method to 'even'.")
    
    tensors = []
    for i in range(1, L+1):
        if index_class==i and class_dim is not None:
            shape = generate_shape(shape_method, L, bond_dim, phys_dim, cyclic, index_class, class_dim=class_dim)
        else:
            shape = generate_shape(shape_method, L, bond_dim, phys_dim, cyclic, i, None)

        tensor = initializer(key, shape, dtype)
        tensors.append(jnp.squeeze(tensor))

    if insert and insert < L and shape_method == 'even':
        # does insert have to be 0? TODO - check!
        tensors[insert] /= np.sqrt(phys_dim)
    
    mps = MatrixProductState(tensors, **kwargs)

    if compress:
        if shape_method == 'even':
            mps.compress(form="flat", max_bond=bond_dim)  # limit bond_dim
        else:
            raise ValueError('')
        
    if canonical_center == None:
        norm = mps.norm()
        for tensor in mps.tensors:
                tensor.modify(data=tensor.data / a.do("power", norm, 1 / L))
    else:
        mps.canonize(canonical_center)
        mps.normalize(insert = canonical_center)
    
    return mps