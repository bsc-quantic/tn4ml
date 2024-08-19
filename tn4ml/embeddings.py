import abc
import itertools
from numbers import Number
from typing import Collection, Any
import numpy as onp
from autoray import numpy as np
import autoray as a
import jax.numpy as jnp
from jax import lax
import quimb.tensor as qtn

class Embedding:
    """Data embedding (feature map) class.

    Attributes
    ----------
        dype: :class:`numpy.dype`
            Data Type
    """
    def __init__(self, dtype=onp.float32):
        self.dtype = dtype

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """ Mapping dimension """
        pass

    @property
    @abc.abstractmethod
    def input_dim(self) -> int:
        """ Dimensionality of input feature. 1 = number, 2 = vector """
        pass

    @abc.abstractmethod
    def __call__(self, x: Number) -> jnp.ndarray:
        pass

class trigonometric(Embedding):
    """ Trigonometric feature map.
    
    :math:`\\phi(x_\\textit{j}) = \\left[ cos(\\frac{\\pi}{2}x_\\textit{j}), sin(\\frac{\pi}{2}x_\\textit{j}) \\right]`
    
    Attributes
    ----------
    k: int
        Custom parameter = ``dim/2``.
    """

    def __init__(self, k: int = 1, **kwargs):
        assert k >= 1

        self.k = k
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """ Mapping dimension """
        return self.k * 2

    @property
    def input_dim(self) -> int:
        return 1
    
    def __call__(self, x: Number) -> jnp.ndarray:
        """Embedding function for trigonometric.
        
        Parameters
        ----------
        x: :class:`Number`
            Input feature.
        
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """
        return 1 / jnp.sqrt(self.k) * jnp.array([f((onp.pi * x / 2**i)) for f, i in itertools.product([jnp.cos, jnp.sin], range(1, self.k + 1))])

class fourier(Embedding):
    """ Fourier feature map.
    :math:`\\phi(x_\\textit{j}) = \\frac{1}{\\sqrt{k}}\\left[ cos(\\frac{\\pi x_\\textit{j}}{2}), sin(\\frac{\\pi x_\\textit{j}}{2}), ..., cos(\\frac{\\pi x_\\textit{j}}{2^k}), sin(\\frac{\\pi x_\\textit{j}}{2^k})\\right]`
    
    Attributes
    ----------
    p: int
        Mapping dimension.
    """

    def __init__(self, p: int = 2, **kwargs):
        #assert p >= 2
        self.p = p
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """ Mapping dimension """
        return self.p

    @property
    def input_dim(self) -> int:
        return 1

    def __call__(self, x: Number) -> jnp.ndarray:
        """Embedding function for Fourier.
        
        Parameters
        ----------
        x: Number
            Input feature.
        
        Returns
        -------
        jnp.ndarray
            Embedding vector."""
        return 1 / self.p * jnp.array([np.abs(sum((np.exp(1j * 2 * onp.pi * k * ((self.p - 1) * x - j) / self.p) for k in range(self.p)))) for j in range(self.p)])
    

class linear_complement_map(Embedding):
    """Feature map :math:`[x, 1-x]` or :math:`[1, x, 1-x]`
    where x = feature in range [0,1].

    Attributes
    ----------
    p: int
        Mapping dimension.
    """
    def __init__(self, p: int = 2, **kwargs):
        self.p = p
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """ Mapping dimension """
        return self.p

    @property
    def input_dim(self) -> int:
        return 1

    def __call__(self, x: Number) -> jnp.ndarray:
        """Embedding function for original inverse.
        
        Parameters
        ----------
        x: :class:`Number`
            Input feature.
        
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """
        if self.p == 2:
            vector = jnp.asarray([x, 1.0 - x])
        elif self.p == 3:
            vector = jnp.asarray([1.0, x, 1.0 - x])
        else:
            raise ValueError('Invalid dimension')
        return vector / jnp.linalg.norm(vector)
    
class quantum_basis(Embedding):
    # fix that it works for any dimension
    """ Basis quantum feature map.  
    The basis is a dictionary of quantum states.
    
    Attributes
    ----------
    basis: :class:`numpy.ndarray`
        quantum state map. Example: {0: [1, 0], 1: [0, 1]}
    """
    def __init__(self, basis: dict, **kwargs):
        self.basis = basis
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """ Mapping dimension """
        return len(self.basis.keys())

    @property
    def input_dim(self) -> int:
        return 1

    def __call__(self, x: Number) -> jnp.ndarray:
        """Embedding function for basis encoding.
        
        Parameters
        ----------
        x: Number
            Input feature.
        
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """
        true_fun = lambda _: jnp.array(self.basis[0])
        false_fun = lambda _: jnp.array(self.basis[1])
        return lax.cond(x == 0, true_fun, false_fun, None)


class gaussian_rbf(Embedding):
    """ Gaussian Radial Basis Function.
    
    Attributes
    ----------
    centers:  :class:`numpy.ndarray`
        Gaussian centers.
    gamma: float
        Scaling factor
        :math:`\\gamma=\\frac{1}{2\\sigma^2}`
    """

    def __init__(self, centers: onp.ndarray = None , gamma: float = None, **kwargs):
        self.centers = centers
        self.gamma = gamma
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """Mapping dimension"""
        return np.prod(self.centers.shape)
    
    @property
    def input_dim(self) -> int:
        """Dimensionality of input feature. 1 = number"""
        return 1
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embedding function for Gaussian RBF.
        
        Parameters
        ----------
        x : Number
            Input feature.
        
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """
        vector = jnp.exp(-self.gamma*jnp.subtract(x, jnp.array(self.centers)))
        return vector / jnp.linalg.norm(vector)

class polynomial(Embedding):
    """ Polynomial feature map
    
    Attributes
    ----------
    degree : int
        Degree of polynomial.
    """

    def __init__(self, degree: int, **kwargs):
        self.degree = degree
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """ Mapping dimension """
        return self.degree + 1

    @property
    def input_dim(self) -> int:
        """ Dimensionality of input feature. 1 = number"""
        return 1
    
    def __call__(self, x: Number) -> jnp.ndarray:
        """Embedding function for polynomial.
        
        Parameters
        ----------
        x : Number
            Input feature.
        
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """
        return jnp.array([x**i for i in range(self.degree + 1)])
 
class jax_arrays(Embedding):
    """Input arrays to JAX arrays.
    
    Attributes
    ----------
    dim: int
        Dimension of input 
    """
    def __init__(self, dim: int = None, **kwargs):
        super().__init__(**kwargs)
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim
    
    def __call__(self, x: Any) -> jnp.ndarray:
        """Embedding function for JAX arrays.
        
        Parameters
        ----------
        x: Number
            Input feature.
        
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """
        return jnp.array(x)

class add_ones(Embedding):
    def __init__(self, **kwargs):
        """Constructor

        """        
        super().__init__(**kwargs)
        self._dim = 2

    @property
    def dim(self) -> int:
        return self._dim

    def __call__(self, x: Number) -> jnp.ndarray:
        return jnp.array([1.0, x])


def embed(x: onp.ndarray, phi: Embedding = trigonometric(), **mps_opts):
    """Creates a product state from a vector of features `x`.
    Works only if features are separated and not correlated (this check you need to do yourself).

    Parameters
    ----------
    x: :class:`numpy.ndarray`
        Vector of features.
    phi: :class:`tn4ml.embeddings.Embedding`
        Feature map for each feature.
    mps_opts: optional
        Additional arguments passed to MatrixProductState class.
    """
    arrays = [phi(xi).reshape((1, 1, phi.dim)) for xi in x]
    for i in [0, -1]:
        arrays[i] = arrays[i].reshape((1, phi.dim))

    mps = qtn.MatrixProductState(arrays, **mps_opts)
    
    # normalize
    norm = mps.norm()
    for tensor in mps.tensors:
        tensor.modify(data=tensor.data / a.do("power", norm, 1 / len(mps.tensors)))
    return mps