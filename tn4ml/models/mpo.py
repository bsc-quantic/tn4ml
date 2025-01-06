from typing import Any, Tuple
import numpy as np
import autoray as a

from quimb import *
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductOperator

from jax.nn.initializers import Initializer
import jax.numpy as jnp

from .model import Model

class MatrixProductOperator(Model, qtn.MatrixProductOperator):
    """A Trainable MatrixProductOperator class.
    See :class:`quimb.tensor.tensor_1d.MatrixProductOperator` for explanation of other attributes and methods.
    """

    def __init__(self, arrays, **kwargs):
        # if isinstance(arrays, MatrixProductState):
        #     Model.__init__(self)
        #     return
        Model.__init__(self)
        qtn.MatrixProductOperator.__init__(self, arrays, **kwargs)
    
    def normalize(self, insert=None):
        """Function for normalizing tensors of :class:`tn4ml.models.mpo.MatrixProductOperator`.

        Parameters
        ----------
        insert : int
            Index of tensor divided by norm. *Default = None*. When `None` the norm division is distributed across all tensors.
        """
        norm = self.norm()
        if insert == None:
            for tensor in self.tensors:
                tensor.modify(data=tensor.data / a.do("power", norm, 1 / self.L))
        else:
            self.tensors[insert].modify(data=self.tensors[insert].data / norm)
    
    # def copy(self):
    #     """Copies the model.
        
    #     Returns
    #     -------
    #     Model of the same type.
    #     """

    #     model = self.copy()
    #     for key in self.__dict__.keys():
    #         model.__dict__[key] = self.__dict__[key]
    #     return model
    
def trainable_wrapper(mps: qtn.MatrixProductOperator, **kwargs) -> MatrixProductOperator:
    """ Creates a wrapper around qtn.MatrixProductOperator so it can be trainable.

    Parameters
    ----------
    mps : :class:`quimb.tensor.MatrixProductOperator`
        Matrix Product Operator to be trained.

    Returns
    -------
    :class:`tn4ml.models.mps.MatrixProductOperator`
    """
    tensors = mps.arrays
    return MatrixProductOperator(tensors, **kwargs)
    
def generate_shape(method: str,
                    L: int,
                    bond_dim: int = 2,
                    phys_dim: Tuple[int, int] = (2, 2),
                    cyclic: bool = False,
                    position: int = None,
                    ) -> tuple:
    """Returns a shape of tensor.

    Parameters
    ----------
    method : str
        Method on how to create shapes of tensors.  
        'even' = exact dimensions as given by parameters, anything else = truncated dimensions.
    L : int
        Number of tensors.
    bond_dim : int
        Dimension of virtual indices between tensors. *Default = 4*.
    phys_dim :  tuple(int, int)
        Dimension of physical indices for individual tensor - *up* and *down*.
    cyclic : bool
        Flag for indicating if MatrixProductOperator this tensor is part of is cyclic. *Default=False*.
    position : int
        Position of tensor in MatrixProductOperator.
    Returns
    -------
        tuple
    """
    
    if method == 'even':
        shape = (bond_dim, bond_dim, *phys_dim)
        if position == 1:
            shape = (1, bond_dim, *phys_dim)
        if position == L:
            shape = (bond_dim, 1, *phys_dim)
    else:
        # not sure is this needed if I can use compress
        assert not cyclic
        if position > L // 2:
            j = (L + 1 - abs(2*position - L - 1)) // 2
        else:
            j = position
        
        chir = min(bond_dim, phys_dim[0]**j * phys_dim[1]**j)
        chil = min(bond_dim, phys_dim[0]**(j-1) * phys_dim[1] ** (j-1))

        if position > L // 2:
            (chil, chir) = (chir, chil)

        if position == 1:
                shape = (chir, *phys_dim)
        elif position == L:
            shape = (chil, *phys_dim)
        else:
            shape = (chil, chir, *phys_dim)
    
    return shape

def MPO_initialize(L: int,
            initializer: Initializer,
            key: Any,
            dtype: Any = jnp.float_,
            shape_method: str = 'even',
            bond_dim: int = 4,
            phys_dim: Tuple[int, int] = (2, 2),
            add_identity: bool = False,
            boundary: str = 'obc',
            cyclic: bool = False,
            compress: bool = False,
            insert: int = None,
            canonical_center: int = None,
            **kwargs):
    
    """Generates :class:`tn4ml.models.mps.MatrixProductOperator`.

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
    phys_dim :  tuple(int, int)
        Dimension of physical indices for individual tensor - *up* and *down*.
    add_identity : bool
        Flag for adding identity to tensor diagonal elements. *Default = False*.
    boundary : str
        Boundary conditions for the MatrixProductOperator. *Default = 'obc'*.
        obc = open boundary conditions. pbc = periodic boundary conditions.
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
    :class:`tn4ml.models.mps.MatrixProductOperator`
    """

    if cyclic and shape_method != 'even':
        raise NotImplementedError("Change shape_method to 'even'.")
    
    tensors = []
    for i in range(1, L+1):
        shape = generate_shape(shape_method, L, bond_dim, phys_dim, cyclic, i)
        
        tensor = initializer(key, shape, dtype)

        if add_identity:
            if len(tensor.shape) == 3:
                copy_tensor = jnp.copy(tensor)
                copy_tensor.at[:, :, 0].set(jnp.eye(tensor.shape[0],
                                                tensor.shape[1],
                                                dtype=dtype))
                tensor = copy_tensor
            elif len(tensor.shape) == 4: # output node
                copy_tensor = jnp.copy(tensor)
                identity = jnp.eye(tensor.shape[0],
                                tensor.shape[1],
                                dtype=dtype)
                identity = jnp.expand_dims(identity, axis=2)
                identity = jnp.broadcast_to(identity, (copy_tensor.shape[0], copy_tensor.shape[1], copy_tensor.shape[3]))
                copy_tensor.at[:, :, 0, :].set(identity)
                tensor = copy_tensor
        
        if boundary == 'obc':
            aux_tensor = jnp.zeros(tensor.shape, dtype=dtype)
            if len(tensor.shape) == 3:
                if i == 1:
                    # Left node
                    aux_tensor = aux_tensor.at[:,0,:].set(tensor[:,0,:])
                    tensor = aux_tensor
                elif i == L:
                    # Right node
                    aux_tensor = aux_tensor.at[0,:,:].set(tensor[0,:,:])
                    tensor = aux_tensor
            elif len(tensor.shape) == 4:
                if i == 1:
                    # Left node
                    aux_tensor = aux_tensor.at[:,0,:,:].set(tensor[:,0,:,:])
                    tensor = aux_tensor
                elif i == L:
                    # Right node
                    aux_tensor = aux_tensor.at[0,:,:,:].set(tensor[0,:,:,:])
                    tensor = aux_tensor
        tensors.append(tensor)

    if insert and insert < L and shape_method == 'even':
        tensors[insert] /= np.sqrt(min(bond_dim, phys_dim[0]))
    
    mpo = MatrixProductOperator(tensors, **kwargs)
    
    if compress and shape_method == 'even':
        mpo.compress(form="flat", max_bond=bond_dim)  # limit bond_dim

    if canonical_center == None:
        mpo.normalize()
    else:
        mpo.canonicalize(canonical_center, inplace=True)
        mpo.normalize(insert = canonical_center)
    
    return mpo