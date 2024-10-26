from typing import Any
import numpy as np
import jax
from jax._src import core, dtypes
from jax import random
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.nn.initializers import *
from .util import gramschmidt_col, gramschmidt_row

def zeros(std: Any = 1e-9, 
            dtype: Any = jnp.float_) -> Initializer:
    """Builds an initializer that initializes tensors with zeros. Plus small noise.

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
        return jax.nn.initializers.zeros(key, shape, dtype) + std * random.normal(key, shape, dtype)
    return init

def ones(std: Any = 1e-9, 
              dtype: Any = jnp.float_) -> Initializer:
    """Builds an initializer that initializes tensors with ones. Plus small noise.

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
        return jax.nn.initializers.ones(key, shape, dtype) + std * random.normal(key, shape, dtype)
    return init

def gramschmidt(dist: str,
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

def identity(type: str,
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

        # Add random noise
        if std:
            tensor += std * random.normal(key, shape, dtype)
        return tensor
    return init

def randn(std: Any = 1.0,
        mean: Any = 0.0,
        noise_std: Any = None,
        noise_mean: Any = None,
        dtype: Any = jnp.float_
        ) -> Initializer:
    """Builds an initializer that initializes tensor values with normal distribution.

    Parameters
    ----------
        std : Any (Optional). Default = `1.0`.
            Additional noise
        mean : Any (Optional). Default = `0.0`.
            Mean of the normal distribution.
        noise_std : Any (Optional). Default = `None`.
            The standard deviation of the noise distribution (normal).
        noise_mean : Any (Optional). Default = `None`.
            The mean of the noise distribution (normal).
        dtype : Any (Optional). Default = `jnp.float_`.
            The initializer's default dtype.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from tn4ml.initializers import randn_init
    >>> initializer = randn(1e-2)
    >>> initializer(jax.random.key(42), (2, 2), jnp.float32)
    Array([[ 0.00186935,  0.01065333],
            [-0.01559313, -0.01535296]], dtype=float32)
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

        tensor = random.normal(key, shape, dtype)
        tensor = mean + tensor*std

        if noise_std and noise_mean:
            noise = random.normal(key, shape, dtype)
            tensor += noise_mean + noise*noise_std

        return tensor
    return init


def unitary_matrix(key: Any,
                   shape: core.Shape,
                   dtype: Any = jnp.float_) -> jnp.ndarray:
    """
    - from @joserapa98/tensorkrowch -
    Generates random unitary matrix from the Haar measure of size n x n.
    
    Unitary matrix is created as described in this `paper
    <https://arxiv.org/abs/math-ph/0609050v2>`_.

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
            Random unitary matrix.
    """
    assert shape[0] == shape[1], "Matrix should be square!"

    mat = jax.random.normal(key, shape, dtype)
    q, r = jnp.linalg.qr(mat)
    d = jnp.diagonal(r)
    ph = d / jnp.abs(d)
    q = q @ jnp.diag(ph)
    return q

def rand_unitary(dtype: Any = jnp.float_) -> Initializer:
    """Builds an initializer that initializes tensor with stack of random unitary matrices.
    
    Parameters
    ----------
        dtype : Any (Optional). Default = `jnp.float_`.
            The initializer's default dtype.
    
    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from tn4ml.initializers import rand_unitary
    >>> initializer = rand_unitary()
    >>> initializer(jax.random.key(42), (2, 2), jnp.float32)
    Array([[ 0.11903083,  0.99289054],
            [-0.99289054,  0.11903088]], dtype=float32)
    >>> tensor = initializer(jax.random.key(42), (2, 2), jnp.float32)
    >>> jnp.allclose(tensor @ tensor.T.conj(), jnp.eye(2), atol=1e-6)
    True
    
    """
    def init(key: Any,
            shape: core.Shape,
            dtype: Any = dtype) -> jnp.ndarray:
        """
        Initializes a tensor.
        
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
        
        size = max(shape[0], shape[1], shape[2])
        size_1 = min(shape[0], size)
        size_2 = min(shape[1], size)
        
        if len(shape) == 3:
            units = []
            for _ in range(shape[2]):
                tensor = unitary_matrix(key, (size, size), dtype)
                tensor = tensor[:size_1, :size_2]
                units.append(tensor)
            tensor = jnp.stack(units, axis=-1)
        elif len(shape) == 4:
            units = []
            for _ in range(shape[-2]):
                inner_units = []
                for _ in range(shape[-1]):
                    unitary = unitary_matrix(key, (size, size), dtype)
                    unitary = unitary[:size_1, :size_2]
                    inner_units.append(unitary)
                inner_stack = jnp.stack(inner_units, axis=-1)
                units.append(inner_stack)
            tensor = jnp.stack(units, axis=-1)
        else:
            raise ValueError("Only 3 and 4 rank tensors are supported!")
        return tensor
    return init

