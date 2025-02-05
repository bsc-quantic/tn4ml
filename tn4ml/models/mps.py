from typing import Any, Collection
import numpy as np
import autoray as a
import math

from quimb import *
import quimb.tensor as qtn

from jax.nn.initializers import Initializer
import jax.numpy as jnp
import jax

from .model import Model
from .tn import TensorNetwork
from ..initializers import randn, rand_unitary

class MatrixProductState(Model, qtn.MatrixProductState):
    """A Trainable MatrixProductState class.
    See :class:`quimb.tensor.tensor_1d.MatrixProductState` for explanation of other attributes and methods.
    """

    def __init__(self,
                    arrays,
                    **kwargs):
        """Initializes the MatrixProductState.

        Parameters
        ----------
        arrays : list of array_like
            The list of tensors, each of shape ``(D, D, d)``, where ``D`` is the bond dimension and ``d`` is the physical dimension.
        **kwargs : dict
            Additional arguments to be passed to the parent class.
        """

        
        Model.__init__(self)
        qtn.MatrixProductState.__init__(self, arrays, **kwargs)
        
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
                    class_index: int = None,
                    class_dim: int = None,
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
    class_index : int
        Index of tensor that is the output node. For classification tasks only.
    class_dim : int
        Dimension of output node, or number of classes for classification.
    Returns
    -------
        tuple
    """

    if method == 'even':
        shape = (bond_dim, bond_dim, phys_dim, class_dim) if class_index is not None and position == class_index else (bond_dim, bond_dim, phys_dim)
        if position == 1:
            shape = (1, bond_dim, phys_dim, class_dim) if class_index is not None and position == class_index else (1, bond_dim, phys_dim)
        if position == L:
            shape = (bond_dim, 1, phys_dim, class_dim) if class_index is not None and position == class_index else (bond_dim, 1, phys_dim)
    else:
        assert not cyclic
        j = (L + 1 - abs(2*position - L - 1)) // 2 if position > L // 2 else position
        
        chir = min(bond_dim, phys_dim**j)
        chil = min(bond_dim, phys_dim**(j-1))
        
        if position > L // 2:
            (chil, chir) = (chir, chil)
        
        if position == 1:
            shape = (1, chir, phys_dim, class_dim) if class_index is not None and position == class_index else (1, chir, phys_dim)
        elif position == L:
            shape = (chil, 1, phys_dim, class_dim) if class_index is not None and position == class_index else (chil, 1, phys_dim)
        else:
            shape = (chil, chir, phys_dim, class_dim) if class_index is not None and position == class_index else (chil, chir, phys_dim)
    
    return shape

def generate_ind(L: int, shape: tuple, position: int, cyclic: bool = False, class_index: int = None) -> tuple:
    """
    Returns the names of the tensor indices.

    Parameters
    ----------
    shape : tuple
        Shape of tensor.
    position : int
        Position of tensor in MatrixProductState. Goes from 1 to L included.
    cyclic : bool
        Flag for indicating if MatrixProductState this tensor is part of is cyclic. *Default=False*.
    class_index : int
        Index of tensor that is the output node (that is having index for number of classes). For classification tasks only.
    
    Returns
    -------
    tuple
        String names of indices.

    """
    if len(shape) == 3:
        if position == 1:
            if class_index == position:
                ind = (f'bond_{position-1}', f'k{position-1}', f'b_{position-1}')
            else:
                ind = (f'bond_{position-2}', f'bond_{position-1}', f'k{position-1}')
        elif position == L:
            if cyclic and class_index != position:
                raise ValueError('Cyclic MPS cannot have class_dim')
            ind = (f'bond_{position-2}', f'k{position-1}', f'b_{position-1}') if class_index == position else (f'bond_{position-2}', f'bond_{position-1}', f'k{position-1}')
        else:
            ind = (f'bond_{position-2}', f'bond_{position-1}', f'k{position-1}')
    else:
        ind = (f'bond_{position-2}', f'bond_{position-1}', f'k{position-1}', f'b_{position-1}')

    return ind

def MPS_initialize(L: int,
                arrays: list = None,
                initializer: Initializer = None,
                key: Any = None,
                dtype: Any = jnp.float_,
                shape_method: str = 'even',
                bond_dim: int = 4,
                phys_dim: int = 2,
                cyclic: bool = False,
                add_identity: bool = False,
                add_to_output: bool = False,
                boundary: str = 'obc',
                class_index: int = None,
                class_dim: int = None,
                tags_id: str = 'I{}',
                compress: bool = False,
                insert: int = None,
                canonical_center: int = None,
                **kwargs):
        
        """Initializes :class:`tn4ml.models.mps.MatrixProductState`.

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
        add_identity : bool
            Flag to add identity to tensors diagonal elements.
        add_to_output : bool
            Flag for adding identity to diagonal elements of tensors with output indices. *Default=False*.
        boundary : str
            Boundary condition of MatrixProductState. *Default = 'obc'*. obc = open boundary condition. pbc = periodic boundary condition.
        class_index : int
            Index of tensor that is the output node for class. For classification tasks only.
        class_dim : int
            Dimension of output node, or number of classes for classification.
        compress : bool
            Flag to truncate bond dimensions.
        insert : int
            Index of tensor divided by norm. When `None` the norm division is distributed across all tensors
        canonical_center : int
            If not `None` then create canonical form around canonical center index.
        kwargs : dict
            Additional arguments.

        Returns
        -------
        :class:`tn4ml.models.mps.MatrixProductState`
        """

        if cyclic and shape_method != 'even':
            raise NotImplementedError("Change shape_method to 'even'.")
        
        if initializer is not None and callable(initializer) and 'rand_unitary' in initializer.__qualname__:
            if add_identity:
                raise ValueError("rand_unitary initializer does not support add_identity.")
            if compress:
                raise ValueError("rand_unitary initializer does not support compress.")
            if insert:
                raise ValueError("rand_unitary initializer does not support insert.")
            if canonical_center:
                raise ValueError("rand_unitary initializer does not support canonization.")
            if boundary == 'obc':
                boundary = None
        
        if arrays is not None:
            # This means MPS for classification needs to be created with qtn.tensor_1d.TensorNetwork1DFlat class
            assert class_index is not None # class_index is required when arrays or shapes are provided
        
        if initializer is None:
            initializer = randn()

        if class_index is not None:
            # MPS for classification
            if class_index > L:
                raise ValueError("class_index should be less than L.")

            tensors = []
            if arrays is not None:
                
                for i, array in enumerate(arrays):
                    ind = generate_ind(L, array.shape, i+1, cyclic, class_index)
                    tensors.append(qtn.Tensor(array,
                                                inds=ind,
                                                tags=tags_id.format(i)))
            else:
                for i in range(1, L+1):
                    shape = generate_shape(shape_method, L, bond_dim, phys_dim, cyclic, i, class_index, class_dim)
                    ind = generate_ind(L, shape, i, cyclic, class_index)

                    if callable(initializer) and 'rand_unitary' in initializer.__qualname__:
                        if i < class_index or i > class_index:
                            array = initializer(key, shape, dtype)
                        elif i == class_index: 
                            # Output node
                            array = jnp.asarray(np.random.normal(0., 1., shape), dtype)
                        else:
                            raise ValueError("Check value of class_index. It should be less than L.")
                    else:
                        array = initializer(key, shape, dtype)
                
                        if add_identity:
                            if len(array.shape) == 3:
                                copy_array = jnp.copy(array)
                                copy_array = copy_array.at[:, :, 0].add(jnp.eye(array.shape[0],
                                                                array.shape[1],
                                                                dtype=dtype))

                                array = copy_array
                            elif len(array.shape) == 4: # output node
                                if add_to_output:
                                    copy_array = jnp.copy(array)
                                    identity = jnp.eye(array.shape[0],
                                                        array.shape[1],
                                                        dtype=dtype)
                                    identity = jnp.expand_dims(identity, axis=2)
                                    identity = jnp.broadcast_to(identity, (copy_array.shape[0], copy_array.shape[1], copy_array.shape[3]))
                                    copy_array = copy_array.at[:, :, 0, :].add(identity)
                                    array = copy_array
                            else:
                                raise ValueError("Tensors need to always be 3D or 4D in MPS for classification.")
                        
                        if boundary == 'obc':
                            aux_array = jnp.zeros(array.shape, dtype=dtype)
                            if i == 1:
                                # Left node
                                aux_array = aux_array.at[:,0,:].set(array[:,0,:])
                                array = aux_array
                            elif i == L:
                                # Right node
                                aux_array = aux_array.at[0,:,:].set(array[0,:,:])
                                array = aux_array
                    tensors.append(qtn.Tensor(array,
                                    inds=ind,
                                    tags=tags_id.format(i-1)))
            
            mps = TensorNetwork(tensors, cyclic=cyclic, site_tag_id=tags_id, **kwargs)

            # normalize
            if canonical_center is None:
                mps.normalize()
            else:
                mps.canonize(canonical_center, inplace=True)
                mps.normalize(insert = canonical_center)
        else:
            # MPS for regression
            if arrays is not None:
                tensors = []
                for i, array in enumerate(arrays):
                    tensors.append(jnp.squeeze(array))
            else:
                tensors = []
                for i in range(1, L+1):
                    shape = generate_shape(shape_method, L, bond_dim, phys_dim, cyclic, i)

                    tensor = initializer(key, shape, dtype)

                    if callable(initializer) and 'rand_unitary' not in initializer.__qualname__:
                        if add_identity:
                            if len(tensor.shape) == 3:
                                copy_tensor = jnp.copy(tensor)
                                copy_tensor.at[:, :, 0].add(jnp.eye(tensor.shape[0],
                                                        tensor.shape[1],
                                                        dtype=dtype))
                                tensor = copy_tensor
                            else:
                                raise ValueError("There was an error in generating shape. They should be 3D")
                            
                        if boundary == 'obc':
                            aux_tensor = jnp.zeros(tensor.shape, dtype=dtype)
                            if i == 1:
                                # Left node
                                aux_tensor = aux_tensor.at[:,0,:].set(tensor[:,0,:])
                                tensor = aux_tensor
                            elif i == L:
                                # Right node
                                aux_tensor = aux_tensor.at[0,:,:].set(tensor[0,:,:])
                                tensor = aux_tensor
                    
                    tensors.append(jnp.squeeze(tensor))
                    
                if not (callable(initializer) and 'rand_unitary' in initializer.__qualname__):
                    if insert and insert < L and shape_method == 'even':
                        tensors[insert] /= jnp.sqrt(phys_dim)
            
            mps = MatrixProductState(tensors, **kwargs)

            if compress:
                if shape_method == 'even':
                    mps.compress(form="flat", max_bond=bond_dim)  # limit bond_dim
                else:
                    raise ValueError('Compress only works with shape_method = "even".')
                
            if canonical_center is None:
                norm = mps.norm()
                for tensor in mps.tensors:
                    tensor.modify(data=tensor.data / a.do("power", norm, 1 / L))
            else:
                mps.canonicalize(canonical_center, inplace=True)
                mps.normalize(insert = canonical_center)
        return mps