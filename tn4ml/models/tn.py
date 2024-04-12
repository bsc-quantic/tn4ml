import copy
from typing import Any, Collection
import numpy as np
import autoray as a

from quimb import *
import quimb.tensor as qtn
from quimb.tensor.tensor_core import TensorNetwork
from jax.nn.initializers import Initializer
import jax.numpy as jnp

from .model import Model

class ParametrizedTensorNetwork(Model, TensorNetwork):
    """A Trainable TensorNetwork class.
    See :class:`quimb.tensor.tensor_core.TensorNetwork` for explanation of other attributes and methods.
    """
    _EXTRA_PROPS = ("_L")
    def __init__(self, tensors, **kwargs):
        """Initializes :class:`tn4ml.models.tn.ParametrizedTensorNetwork`.
        
        Parameters
        ----------
        tensors : list or TensorNetwork
            List of tensors of :class:`quimb.tensor.tensor_core.Tensor` or :class:quimb.tensor.tensor_core.TensorNetwork.
        kwargs : dict
            Additional arguments.
        """
        if isinstance(tensors, TensorNetwork):
            Model.__init__(self)
            return
        Model.__init__(self)
        TensorNetwork.__init__(self, tensors, **kwargs)

        self.L = len(self.tensors)
    
    # doesn't want to train because this function doesnt work - TODO fix this
    def copy(self, virtual: bool=False, deep: bool=False):
        """Copies the model.
        
        Returns
        -------
        Model of the same type.
        """

        if deep:
            return copy.deepcopy(self)
        
        model = self.__class__(self, virtual=virtual)
        for key in self.__dict__.keys():
            model.__dict__[key] = self.__dict__[key]
        return model
    
def trainable_wrapper(tn: qtn.TensorNetwork, **kwargs) -> ParametrizedTensorNetwork:
    """ Creates a wrapper around qtn.TensorNetwork so it can be trainable.

    Parameters
    ----------
    tn : :class:`quimb.tensor.TensorNetwork`
        Tensor Network to be trained.

    Returns
    -------
    :class:`tn4ml.models.tn.ParametrizedTensorNetwork`
    """
    tensors = tn.tensors
    return ParametrizedTensorNetwork(tensors, **kwargs)

def TN_initialize(arrays: list = None, shapes: list = None, key: Any = None, initializer: Initializer = None, inds: Collection[Collection[str]] = None, tags_id: str = 'I{}', dtype: Any = jnp.float_, **kwargs) -> TensorNetwork:
    """Initializes a TensorNetwork.

    Parameters
    ----------
    arrays : list
        List of arrays to be used as tensors. *Default = None*.  
        If None, shapes must be provided.
    shapes : list
        List of shapes of tensors. Each shape should be in LRP(P) format : (left, right, physical)  
        *Default = None*. If None, arrays must be provided.
    key : Any
        Random key for initialization. *Default = None*.
    initializer : from `tn4ml.initializers` or `jax.nn.initializers`
        Initializer for tensors. *Default = None*.  
        If None, tensors are initialized with random values. Only provided if arrays is None.
    inds : sequence of arrays of str
        List of indices for tensors. *Default = None*.  
        Neeeds to be provided because its showing connectivity between tensors.
        Example for TN with 3 tensors:
        >>> inds = [['bond0', 'k0'], ['bond0', 'bond1', 'k2'], ['bond1', 'k3']]
    tags_id : str
        Tag identifier for tensors. *Default = 'I{}'*.  
        The tag identifier should have a single placeholder for tag number.
    dtype : Any
        Data type for tensors. *Default = jnp.float_*.
    kwargs : dict
        Additional arguments.

    Returns
    -------
    :class:`tn4ml.models.tn.ParametrizedTensorNetwork`
    """
    
    if inds is None:
        raise ValueError("Provide indices for tensors - connectivity map between tensors.")

    tensors = []
    if arrays is not None:
        if len(arrays) != len(inds):
            raise ValueError("Number of tensors and indices should be same.")
        
        for i, array in enumerate(arrays):
            tensors.append(qtn.Tensor(array,
                                      inds=inds[i],
                                      tags=tags_id.format(i)))
    elif shapes is not None:
        if len(shapes) != len(inds):
            raise ValueError("Number of tensors and indices should be same.")
        
        for i, shape in enumerate(shapes):
            if initializer is not None:
                array = np.asarray(initializer(key, shape, dtype))
            else:
                array = np.asarray(np.random.normal(0, 1, shape), dtype)
            
            tensors.append(qtn.Tensor(array,
                                      inds=inds[i],
                                      tags=tags_id.format(i)))
    else:
        raise ValueError("Provide either arrays or shapes to create Tensor Network.")

    return ParametrizedTensorNetwork(tensors)
