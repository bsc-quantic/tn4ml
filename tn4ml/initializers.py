from typing import Any
import numpy as np
import jax
from jax._src import core, dtypes
from jax import random
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.nn.initializers import *
from .util import gramschmidt_col, gramschmidt_row

def zeros_init() -> Initializer:
    """Builds an initializer that initializes tensors with zeros.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from tn4ml.initializers import zeros_init
    >>> initializer = zeros_init()
    >>> initializer(jax.random.key(42), (2, 2), jnp.float32)
    Array([[0., 0.],
       [0., 0.]], dtype=float32)
    """
    
    def init(key: Any,
           shape: core.Shape,
           dtype: Any) -> jnp.ndarray:
        """Initializes a tensor.
        
        Parameters
        ----------
            key : Any
                Random key.
            shape : core.Shape
                Shape of the tensor.
            dtype : Any
                Data type of the tensor.
        
        Returns
        -------
            jnp.ndarray
                Initialized tensor.
        """
        return jax.nn.initializers.zeros(key, shape, dtype)
    return init

def ones_init() -> Initializer:
    """Builds an initializer that initializes tensors with ones.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from tn4ml.initializers import ones_init
    >>> initializer = ones_init()
    >>> initializer(jax.random.key(42), (2, 2), jnp.float32)
    Array([[1., 1.],
       [1., 1.]], dtype=float32)
    """
    
    def init(key: Any,
           shape: core.Shape,
           dtype: Any) -> jnp.ndarray:
        """Initializes a tensor.
        
        Parameters
        ----------
            key : Any
                Random key.
            shape : core.Shape
                Shape of the tensor.
            dtype : Any
                Data type of the tensor.
        
        Returns
        -------
            jnp.ndarray
                Initialized tensor.
        """
        return jax.nn.initializers.ones(key, shape, dtype)
    return init

def gramschmidt_init(dist: str,
                    scale: Any = 1e-2,
                    dtype: Any = jnp.float_
                    ) -> Initializer:
    """Builds an initializer that initializes tensors with Gram-Schmidt orthogonalization procedure.
    First, arrays are sampled from uniform or normal distribution (specified by `dist` argument)

    Parameters
    ----------
        dist : str
            Sampling distribution of arrays. Options: `uniform`, `normal`.
        scale : Any (Optional). Default = `1e-2`.
            Scaling factor for the sampled arrays.
        dtype : Any (Optional)
            The initializer's default dtype.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from tn4ml.initializers import gramschmidt_init
    >>> initializer = gramschmidt_init('normal')
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)
    Array([[ 0.35777482,  0.65598017,  0.6645954 ],
       [-0.57674366, -0.40450865,  0.70974606]], dtype=float32)
    """
    def init(key: Any,
           shape: core.Shape,
           dtype: Any = dtype) -> jnp.ndarray:
        """Initializes a tensor.
        
        Parameters
        ----------
            key : Any
                Random key.
            shape : core.Shape
                Shape of the tensor.
            dtype : Any
                Data type of the tensor.
        
        Returns
        -------
            jnp.ndarray
                Initialized tensor.
        """
        dtype = dtypes.canonicalize_dtype(dtype)
        
        matrix_shape = shape[0], np.prod(shape[1:])

        if dist == 'uniform':
            arrays = random.uniform(key, matrix_shape, dtype) * jnp.array(scale, dtype)
        elif dist == 'normal':
            arrays = random.normal(key, matrix_shape, dtype) * jnp.array(scale, dtype)
        else:
            raise ValueError("Sampling only implemented for 'uniform' and 'normal' distributions!")

        arrays = gramschmidt_row(arrays)
        return arrays.reshape(shape)
    return init

def identity_init(type: str,
                std: Any = None,
                dtype: Any = jnp.float_) -> Initializer:
    """Builds an initializer that initializes tensors with identity either on diagonal elements, or in bond dimensions.

    Parameters
    ----------
        type : str. Options: 'copy', 'bond'
            'copy' = diagonal elements, 'bond' = bond dimension elements
        std : Any (Optional)
            Additonal noise
        dtype :  Any (Optional). Default = `jnp.float_`.
            The initializer's default dtype.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from tn4ml.initializers import gramschmidt_init
    >>> initializer = identity_init('copy', 1e-2)
    >>> initializer(jax.random.key(42), (3, 2), jnp.float32)
    Array([[ 1.0061227 ,  0.01122588],
       [ 0.01137332,  0.99187267],
       [-0.00890405,  0.00126231]], dtype=float32)
    """
    def init(key: Any,
            shape: core.Shape,
            dtype: Any = dtype) -> jnp.ndarray:
        """Initializes a tensor.
        
        Parameters
        ----------
            key : Any
                Random key.
            shape : core.Shape
                Shape of the tensor.
            dtype : Any
                Data type of the tensor.
        
        Returns
        -------
            jnp.ndarray
                Initialized tensor.
        """
        dtype = dtypes.canonicalize_dtype(dtype)
        rank = len(shape)

        if type == 'bond':
            tensor = jnp.zeros(shape, dtype=dtype)
            if rank == 4:
                eye_tensor = jnp.eye(shape[0], shape[1]).reshape(shape[0], shape[1], 1, 1)
            elif rank == 3:
                eye_tensor = jnp.eye(shape[0], shape[1]).reshape(shape[0], shape[1], 1)
            else:
                raise ValueError('Tensor should have LRP shape')
            # Use broadcasting to fill tensor slices
            tensor += eye_tensor

        elif type == 'copy':
            # from @joserapa98/tensorkrowch
            tensor = jnp.zeros(shape, dtype=dtype)
            rank = len(shape)
            if rank <= 1:
                i = 0
            else:
                i = np.arange(min(shape), dtype=int)
            tensor = tensor.at[(i,) * rank].set(1.)
        else:
            raise ValueError('Defined only for diagonal and bond dimension identity intialization!')

        # Add on a bit of random noise
        if std:
            tensor += std * random.normal(key, shape, dtype)
        return tensor
    return init

def noise_init(std: Any = 1e-9,
                dtype: Any = jnp.float_
                ) -> Initializer:
    """Builds an initializer that initializes tensor values with normal distribution and added noise.

    Parameters
    ----------
        std : Any (Optional). Default = `1e-9`.
            Additional noise
        dtype : Any (Optional). Default = `jnp.float_`.
            The initializer's default dtype.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from tn4ml.initializers import noise_init
    >>> initializer = noise_init(1e-8)
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)
    Array([[ 6.12265216e-09,  1.12258824e-08,  1.13733174e-08],
       [-8.12732548e-09, -8.90404994e-09,  1.26231448e-09]], dtype=float32)
    """
    def init(key: Any,
            shape: core.Shape,
            dtype: Any = dtype) -> jnp.ndarray:
        """Initializes a tensor.
        
        Parameters
        ----------
            key : Any
                Random key.
            shape : core.Shape
                Shape of the tensor.
            dtype : Any
                Data type of the tensor.
        
        Returns
        -------
            jnp.ndarray
                Initialized tensor.
        """
        dtype = dtypes.canonicalize_dtype(dtype)

        if not std:
            raise ValueError('Provide noise!')
        return std * random.normal(key, shape, dtype)
    return init
   


