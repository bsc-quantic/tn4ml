import copy
from typing import Any, Collection
import numpy as np
import autoray as a

from quimb import *
import quimb.tensor as qtn
from jax.nn.initializers import Initializer
import jax.numpy as jnp

from .model import Model
from ..initializers import *

class TensorNetwork(Model, qtn.tensor_1d.TensorNetwork1DFlat):
    """A Trainable TensorNetwork class.
    See :class:`quimb.tensor.tensor_core.TensorNetwork` for explanation of other attributes and methods.
    """
    _EXTRA_PROPS = ("_L", "_site_tag_id", "cyclic")
    def __init__(self, tensors, site_tag_id:str="I{}", cyclic:bool=False, **kwargs):
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
        qtn.tensor_1d.TensorNetwork1DFlat.__init__(self, tensors, **kwargs)

        self._L = len(self.tensors)
        self.cyclic = cyclic
        self._site_tag_id = site_tag_id
    
    def canonize(self, where, cur_orthog='calc', info=None, bra=None, inplace=False):
        """Canonizes the tensor network.
        """
        self.canonicalize(where, cur_orthog=cur_orthog, info=info, bra=bra, inplace=inplace)

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

    def norm(self, **contract_opts) -> float:
        """Calculates norm of :class:`tn4ml.models.tn.TensorNetwork`.

        Parameters
        ----------
        contract_opts : Optional
            Arguments passed to ``contract()``.

        Returns
        -------
        float
            Norm of :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        """
        norm = self.conj() & self
        return norm.contract(**contract_opts) ** 0.5
    
    def normalize(self, insert=None) -> None:
        """Function for normalizing tensors of :class:`tn4ml.models.tn.TensorNetwork`.

        Parameters
        ----------
        insert : int
            Index of tensor divided by norm. *Default = None*. When `None` the norm division is distributed across all tensors.
        """

        if not self.tensors:
            raise ValueError("The tensor network is empty.")
        
        norm = self.norm()
        
        if insert == None:
            for tensor in self.tensors:
                tensor.modify(data=tensor.data / a.do("power", norm, 1 / self.L))
        else:
            if not (0 <= insert < len(self.tensors)):
                raise IndexError(f"Insert index {insert} is out of bounds for the tensor list.")
            self.tensors[insert].modify(data=self.tensors[insert].data / norm)
    
def trainable_wrapper(tn: qtn.tensor_1d.TensorNetwork1DFlat, **kwargs) -> qtn.tensor_1d.TensorNetwork1DFlat:
    """ Creates a wrapper around qtn.tensor_1d.TensorNetwork1DFlat so it can be trainable.

    Parameters
    ----------
    tn : :class:`quimb.tensor.TensorNetwork`
        Tensor Network to be trained.

    Returns
    -------
    :class:`tn4ml.models.tn.TensorNetwork`
    """
    tensors = tn.tensors
    return TensorNetwork(tensors, **kwargs)

def TN_initialize(arrays: list = None,
                  shapes: list = None, 
                  key: Any = None,
                  initializer: Initializer = None,
                  inds: Collection[Collection[str]] = None,
                  tags_id: str = 'I{}',
                  cyclic: bool = False,
                  dtype: Any = jnp.float_,
                  **kwargs) -> TensorNetwork:
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
    :class:`tn4ml.models.tn.TensorNetwork`
    """

    if arrays is None and shapes is None:
        raise ValueError("Provide either arrays or shapes to create Tensor Network.")
    
    L = len(arrays) if arrays is not None else len(shapes)

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
        
        for i, shape in zip(range(1, L+1), shapes):
            
            if initializer is not None:
                array = initializer(key, shape, dtype)
            else:
                array = np.asarray(np.random.normal(0., 1., shape), dtype)
            
            tensors.append(qtn.Tensor(array,
                                      inds=inds[i-1],
                                      tags=tags_id.format(i-1)))
    
    tn = TensorNetwork(tensors, cyclic=cyclic, site_tag_id=tags_id, **kwargs)

    # normalize
    tn.normalize()

    return tn
