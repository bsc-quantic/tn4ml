import abc
import itertools
import math
from numbers import Number
from typing import Collection, Any, Union, Optional, Dict, List, Tuple

import numpy as onp
import jax
import jax.numpy as jnp
from jax import lax
import quimb.tensor as qtn

import tn4ml.util as u
from tn4ml.scipy.special import eval_legendre, eval_laguerre, eval_hermite

class Embedding(abc.ABC):
    """Base class for data embeddings (feature maps).
    
    This abstract base class defines the interface for all embedding implementations.
    Each embedding maps input data to a higher dimensional space for tensor network operations.
    
    Attributes
    ----------
    dtype : :class:`numpy.dtype`
        Data type for computations. Defaults to float32.
    """
    
    def __init__(self, dtype: onp.dtype = onp.float32):
        """Initialize the embedding.
        
        Parameters
        ----------
        dtype : :class:`numpy.dtype`, optional
            Data type for computations, by default float32
        """
        self.dtype = dtype

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Get the output dimension of the embedding.
        
        Returns
        -------
        int
            The dimension of the output vector
        """
        pass

    @property
    @abc.abstractmethod
    def input_dim(self) -> int:
        """Get the input dimension of the embedding.
        
        Returns
        -------
        int
            The dimension of the input (1 for scalar, 2 for vector)
        """
        pass

    @abc.abstractmethod
    def __call__(self, x: Number) -> jnp.ndarray:
        """Apply the embedding to input data.
        
        Parameters
        ----------
        x : Number
            Input data to embed
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Embedded vector
        """
        pass

class ComplexEmbedding(abc.ABC):
    """Base class for complex embeddings with multiple feature dimensions.
    
    This abstract base class extends Embedding to handle multiple features,
    each with its own embedding function.
    
    Attributes
    ----------
    dtype : :class:`numpy.dtype`
        Data type for computations. Defaults to float32.
    """
    
    def __init__(self, dtype: onp.dtype = onp.float32):
        """Initialize the complex embedding.
        
        Parameters
        ----------
        dtype : :class:`numpy.dtype`, optional
            Data type for computations, by default float32
        """
        self.dtype = dtype

    @property
    @abc.abstractmethod
    def dims(self) -> Collection[int]:
        """Get the output dimensions for each feature.
        
        Returns
        -------
        Collection[int]
            List of dimensions for each feature's output
        """
        pass

    @property
    @abc.abstractmethod
    def input_dims(self) -> jnp.ndarray:
        """Get the input dimensions for each feature.
        
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Array of input dimensions (1 for scalar, 2 for vector)
        """
        pass

    @property
    @abc.abstractmethod
    def embeddings(self) -> Collection[Embedding]:
        """Get the embedding functions for each feature.
        
        Returns
        -------
        Collection[Embedding]
            List of embedding functions
        """
        pass

    @abc.abstractmethod
    def __call__(self, x: Number) -> jnp.ndarray:
        """Apply the complex embedding to input data.
        
        Parameters
        ----------
        x : Number
            Input data to embed
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Embedded vector
        """
        pass

class StateVectorToMPSEmbedding(abc.ABC):
    """Base class for converting state vectors to Matrix Product States (MPS).
    
    This abstract base class provides functionality to convert quantum state vectors
    into MPS representation.
    
    Attributes
    ----------
    dtype : :class:`numpy.dtype`
        Data type for computations
    max_bond : Optional[int]
        Maximum bond dimension for MPS decomposition
    """
    
    def __init__(self, dtype: onp.dtype = onp.float32, max_bond: Optional[int] = None):
        """Initialize the state vector to MPS embedding.
        
        Parameters
        ----------
        dtype : :class:`numpy.dtype`, optional
            Data type for computations, by default float32
        max_bond : Optional[int], optional
            Maximum bond dimension for MPS decomposition, by default None
        """
        self.dtype = dtype
        self.max_bond = max_bond

    @property
    @abc.abstractmethod
    def dims(self) -> list:
        """Get dimensions of the MPS tensors.
        
        Returns
        -------
        list
            List of tensor shapes
        """
        pass

    @property
    @abc.abstractmethod
    def create_statevector(self, x: jnp.ndarray) -> jnp.ndarray:
        """Create a state vector from input data.
        
        Parameters
        ----------
        x : :class:`jax.numpy.ndarray`
            Input data
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            State vector representation
        """
        pass
    
    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert input data to MPS representation.
        
        Parameters
        ----------
        x : :class:`jax.numpy.ndarray`
            Input data
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            MPS representation
        """
        pass

class MPSEmbedding(abc.ABC):
    """Base class for converting input data to Matrix Product State (MPS).
    
    This abstract base class provides functionality to convert input data
    into MPS representation using custom decomposition strategies.
    
    Attributes
    ----------
    dtype : :class:`numpy.dtype`
        Data type for computations
    max_bond : Optional[int]
        Maximum bond dimension for MPS decomposition
    """
    
    def __init__(self, dtype: onp.dtype = onp.float32, max_bond: Optional[int] = None):
        """Initialize the MPS embedding.
        
        Parameters
        ----------
        dtype : :class:`numpy.dtype`, optional
            Data type for computations, by default float32
        max_bond : Optional[int], optional
            Maximum bond dimension for MPS decomposition, by default None
        """
        self.dtype = dtype
        self.max_bond = max_bond

    @property
    @abc.abstractmethod
    def dims(self) -> list:
        """Get dimensions of the MPS tensors.
        
        Returns
        -------
        list
            List of tensor shapes
        """
        pass

    @property
    @abc.abstractmethod
    def decompose(self, x: Any, *args) -> jnp.ndarray:
        """Decompose input data into MPS format.
        
        Parameters
        ----------
        x : Any
            Input data
        *args : Any
            Additional arguments for decomposition
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Decomposed data in MPS format
        """
        pass
    
    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert input data to MPS representation.
        
        Parameters
        ----------
        x : :class:`jax.numpy.ndarray`
            Input data
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            MPS representation
        """
        pass

class TrigonometricEmbedding(Embedding):
    """TrigonometricEmbedding feature map with multiple frequency components.
    
    Maps input x to :math:`\\phi(x) = \\frac{1}{\\sqrt{k}}[\\cos(\\frac{\\pi}{2}x), \\sin(\\frac{\\pi}{2}x), ..., \\cos(\\frac{\\pi}{2^k}x), \\sin(\\frac{\\pi}{2^k}x)]`
    
    Attributes
    ----------
    k : int
        Number of frequency components (dim/2)
    """
    
    def __init__(self, k: int = 1, **kwargs):
        """Initialize the TrigonometricEmbedding.
        
        Parameters
        ----------
        k : int, optional
            Number of frequency components, by default 1
        **kwargs : Any
            Additional arguments passed to parent class
            
        Raises
        ------
        AssertionError
            If k < 1
        """
        assert k >= 1, "k must be at least 1"
        self.k = k
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """Get the output dimension (2k)."""
        return self.k * 2

    @property
    def input_dim(self) -> int:
        """Get the input dimension (1 for scalar input)."""
        return 1
    
    def __call__(self, x: Number) -> jnp.ndarray:
        """Apply TrigonometricEmbedding to input.
        
        Parameters
        ----------
        x : Number
            Input value
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Embedded vector with cosine and sine components
        """
        return 1 / jnp.sqrt(self.k) * jnp.array([
            f((onp.pi * x / 2**i)) 
            for f, i in itertools.product([jnp.cos, jnp.sin], range(1, self.k + 1))
        ])

class FourierEmbedding(Embedding):
    """Fourier feature map with multiple frequency components.
    
    Maps input x to :math:`\\phi(x) = \\frac{1}{\\sqrt{p}}[\\cos(2\\pi 0 x), ..., \\cos(2\\pi (p-1) x), \\sin(2\\pi 0 x), ..., \\sin(2\\pi (p-1) x)]`
    
    Attributes
    ----------
    p : int
        Number of frequency components
    """
    
    def __init__(self, p: int = 2, **kwargs):
        """Initialize the Fourier embedding.
        
        Parameters
        ----------
        p : int, optional
            Number of frequency components, by default 2
        **kwargs : Any
            Additional arguments passed to parent class
            
        Raises
        ------
        AssertionError
            If p < 1
        """
        assert p >= 1, "Number of frequency components must be at least 1"
        self.p = p
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """Get the output dimension (2p)."""
        return self.p

    @property
    def input_dim(self) -> int:
        """Get the input dimension (1 for scalar input)."""
        return 1

    def __call__(self, x: Number) -> jnp.ndarray:
        """Apply Fourier embedding to input.
        
        Parameters
        ----------
        x : Number
            Input value in [0,1]
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Embedded vector with cosine and sine components
        """
        return 1 / self.p * jnp.array([jnp.abs(sum((jnp.exp(1j * 2 * jnp.pi * k * ((self.p - 1) * x - j) / self.p) for k in range(self.p)))) for j in range(self.p)])

class LinearComplementEmbedding(Embedding):
    """Linear complement feature map.
    
    Maps input x to either [x, 1-x] or [1, x, 1-x] where x is in [0,1].
    
    Attributes
    ----------
    p : int
        Output dimension (2 or 3)
    """
    
    def __init__(self, p: int = 2, **kwargs):
        """Initialize the linear complement embedding.
        
        Parameters
        ----------
        p : int, optional
            Output dimension (2 or 3), by default 2
        **kwargs : Any
            Additional arguments passed to parent class
            
        Raises
        ------
        ValueError
            If p is not 2 or 3
        """
        if p not in [2, 3]:
            raise ValueError('p must be 2 or 3')
        self.p = p
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """Get the output dimension."""
        return self.p

    @property
    def input_dim(self) -> int:
        """Get the input dimension (1 for scalar input)."""
        return 1

    def __call__(self, x: Number) -> jnp.ndarray:
        """Apply linear complement embedding to input.
        
        Parameters
        ----------
        x : Number
            Input value in [0,1]
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Embedded vector [x, 1-x] or [1, x, 1-x]
        """
        if self.p == 2:
            vector = jnp.asarray([x, 1.0 - x])
        else:  # p == 3
            vector = jnp.asarray([1.0, x, 1.0 - x])
        return vector / jnp.linalg.norm(vector)

class QuantumBasisEmbedding(Embedding):
    """Quantum basis feature map using dictionary of quantum states.
    
    Maps input x to quantum states from a predefined basis.
    
    Attributes
    ----------
    basis : Dict[int, List[float]]
        Dictionary mapping input values to quantum states
    """
    
    def __init__(self, basis: Dict[int, List[float]], **kwargs):
        """Initialize the quantum basis embedding.
        
        Parameters
        ----------
        basis : Dict[int, List[float]]
            Dictionary mapping input values to quantum states
        **kwargs : Any
            Additional arguments passed to parent class
        """
        self.basis = basis
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """Get the output dimension (size of basis)."""
        return len(self.basis.keys())

    @property
    def input_dim(self) -> int:
        """Get the input dimension (1 for scalar input)."""
        return 1

    def __call__(self, x: Number) -> jnp.ndarray:
        """Apply quantum basis embedding to input.
        
        Parameters
        ----------
        x : Number
            Input value (0 or 1)
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Corresponding quantum state
        """
        true_fun = lambda _: jnp.array(self.basis[0])
        false_fun = lambda _: jnp.array(self.basis[1])
        return lax.cond(x == 0, true_fun, false_fun, None)

class GaussianRBFEmbedding(Embedding):
    """Gaussian Radial Basis Function embedding.
    
    Maps input x to Gaussian RBF features centered at specified points.
    
    Attributes
    ----------
    centers : onp.ndarray
        Centers for Gaussian RBFs
    gamma : float
        Scaling factor :math:`\\gamma=\\frac{1}{2\\sigma^2}`
    """
    
    def __init__(self, centers: Optional[onp.ndarray] = None, gamma: Optional[float] = None, **kwargs):
        """Initialize the Gaussian RBF embedding.
        
        Parameters
        ----------
        centers : Optional[onp.ndarray], optional
            Centers for Gaussian RBFs, by default None
        gamma : Optional[float], optional
            Scaling factor, by default None
        **kwargs : Any
            Additional arguments passed to parent class
        """
        self.centers = centers
        self.gamma = gamma
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """Get the output dimension (product of centers shape)."""
        return jnp.prod(self.centers.shape)
    
    @property
    def input_dim(self) -> int:
        """Get the input dimension (1 for scalar input)."""
        return 1
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply Gaussian RBF embedding to input.
        
        Parameters
        ----------
        x : :class:`jax.numpy.ndarray`
            Input value
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Gaussian RBF features
        """
        vector = jnp.exp(-self.gamma * jnp.subtract(x, jnp.array(self.centers)))
        return vector / jnp.linalg.norm(vector)

class PolynomialEmbedding(Embedding):
    """PolynomialEmbedding feature map.
    
    Maps input x to PolynomialEmbedding features up to specified degree.
    
    Attributes
    ----------
    degree : int
        Maximum PolynomialEmbedding degree
    n : int
        Number of input features
    include_bias : bool
        Whether to include constant term
    """
    
    def __init__(self, degree: int, n: int, include_bias: bool = False, **kwargs):
        """Initialize the PolynomialEmbedding.
        
        Parameters
        ----------
        degree : int
            Maximum PolynomialEmbedding degree
        n : int
            Number of input features
        include_bias : bool, optional
            Whether to include constant term, by default False
        **kwargs : Any
            Additional arguments passed to parent class
            
        Raises
        ------
        ValueError
            If degree < 1
        """
        if degree < 1:
            raise ValueError("Degree of PolynomialEmbedding must be at least 1.")
        self.degree = degree
        self.n = n
        self.include_bias = include_bias
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """Get the output dimension (sum of combinations)."""
        if self.include_bias:
            return sum(math.comb(self.input_dim + k - 1, k) for k in range(0, self.degree + 1))
        return sum(math.comb(self.input_dim + k - 1, k) for k in range(1, self.degree + 1))

    @property
    def input_dim(self) -> int:
        """Get the input dimension."""
        return self.n
    
    def __call__(self, x: Union[Number, onp.array]) -> jnp.ndarray:
        """Apply PolynomialEmbedding embedding to input.
        
        Parameters
        ----------
        x : Union[:class:`jax.numpy.ndarray`, :class:`numpy.ndarray`]
            Input features
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            PolynomialEmbedding features
        """
        if x.ndim == 0:
            x = jnp.array([x])
        
        features = []
        if self.include_bias:
            features.append(1.0)        
        
        for d in range(1, self.degree + 1):
            for combination in itertools.combinations_with_replacement(range(len(x)), d):
                product = jnp.prod(x[jnp.array(combination)])
                features.append(product)
        
        return jnp.array(features)

class LegendreEmbedding(Embedding):
    """Legendre PolynomialEmbedding feature map.
    
    Maps input x to Legendre PolynomialEmbedding features.
    
    Attributes
    ----------
    degree : int
        Maximum PolynomialEmbedding degree
    """
    
    def __init__(self, degree: int = 2, **kwargs):
        """Initialize the Legendre embedding.
        
        Parameters
        ----------
        degree : int, optional
            Maximum PolynomialEmbedding degree, by default 2
        **kwargs : Any
            Additional arguments passed to parent class
        """
        self.degree = degree
        super().__init__(**kwargs)
        
    @property
    def dim(self) -> int:
        """Get the output dimension (degree + 1)."""
        return self.degree + 1
        
    @property
    def input_dim(self) -> int:
        """Get the input dimension (1 for scalar input)."""
        return 1
        
    def __call__(self, x: Number) -> jnp.ndarray:
        """Apply Legendre PolynomialEmbedding embedding to input.
        
        Parameters
        ----------
        x : Number
            Input value in [-1, 1]
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Legendre PolynomialEmbedding features
        """
        features = jnp.array([eval_legendre(k, x) for k in range(self.degree + 1)])
        return features
    
class LaguerreEmbedding(Embedding):
    """Laguerre PolynomialEmbedding feature map with isometric weighting.
    
    Maps input x to weighted Laguerre PolynomialEmbedding features.
    
    Attributes
    ----------
    degree : int
        Maximum PolynomialEmbedding degree
    """
    
    def __init__(self, degree: int = 2, **kwargs):
        """Initialize the Laguerre embedding.
        
        Parameters
        ----------
        degree : int, optional
            Maximum PolynomialEmbedding degree, by default 2
        **kwargs : Any
            Additional arguments passed to parent class
        """
        self.degree = degree
        super().__init__(**kwargs)
        
    @property
    def dim(self) -> int:
        """Get the output dimension (degree + 1)."""
        return self.degree + 1
        
    @property
    def input_dim(self) -> int:
        """Get the input dimension (1 for scalar input)."""
        return 1
        
    def __call__(self, x: Number) -> jnp.ndarray:
        """Apply weighted Laguerre PolynomialEmbedding embedding to input.
        
        Parameters
        ----------
        x : Number
            Input value in [0, ∞)
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Weighted Laguerre PolynomialEmbedding features
        """
        weight = jnp.exp(-x / 2)
        features = jnp.array([weight * eval_laguerre(k, x) for k in range(self.degree + 1)])
        return features

class HermiteEmbedding(Embedding):
    """Hermite PolynomialEmbedding feature map with isometric weighting.
    
    Maps input x to weighted Hermite PolynomialEmbedding features.
    
    Attributes
    ----------
    degree : int
        Maximum PolynomialEmbedding degree
    """
    
    def __init__(self, degree: int = 2, **kwargs):
        """Initialize the Hermite embedding.
        
        Parameters
        ----------
        degree : int, optional
            Maximum PolynomialEmbedding degree, by default 2
        **kwargs : Any
            Additional arguments passed to parent class
        """
        self.degree = degree
        super().__init__(**kwargs)
        
    @property
    def dim(self) -> int:
        """Get the output dimension (degree + 1)."""
        return self.degree + 1
        
    @property
    def input_dim(self) -> int:
        """Get the input dimension (1 for scalar input)."""
        return 1
        
    def __call__(self, x: Number) -> jnp.ndarray:
        """Apply weighted Hermite PolynomialEmbedding embedding to input.
        
        Parameters
        ----------
        x : Number
            Input value in R
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Weighted Hermite PolynomialEmbedding features
        """
        weight = jnp.exp(-0.5 * x**2)
        features = jnp.array([weight * eval_hermite(k, x) for k in range(self.degree + 1)])
        return features
 
class JaxArraysEmbedding(Embedding):
    """Simple embedding that converts input arrays to JAX arrays.
    
    Optionally adds a bias term to the input.
    
    Attributes
    ----------
    dim : Optional[int]
        Output dimension
    add_bias : bool
        Whether to add bias term
    input_dim : Optional[int]
        Input dimension
    """
    
    def __init__(self, dim: Optional[int] = None, add_bias: bool = False, input_dim: Optional[int] = None, **kwargs):
        """Initialize the JAX arrays embedding.
        
        Parameters
        ----------
        dim : Optional[int], optional
            Output dimension, by default None
        add_bias : bool, optional
            Whether to add bias term, by default False
        input_dim : Optional[int], optional
            Input dimension, by default None
        **kwargs : Any
            Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self._dim = dim
        self.add_bias = add_bias
        self._input_dim = input_dim

    @property
    def dim(self) -> int:
        """Get the output dimension."""
        return self._dim
    
    @property
    def input_dim(self) -> int:
        """Get the input dimension."""
        return self._input_dim
    
    def __call__(self, x: Any) -> jnp.ndarray:
        """Convert input to JAX array, optionally adding bias.
        
        Parameters
        ----------
        x : Any
            Input data
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            JAX array with optional bias term
        """
        if self.add_bias:
            return jnp.concatenate([jnp.array([1.]), x])
        return jnp.array(x)
    
class TrigonometricEmbeddingChain(ComplexEmbedding):
    """TrigonometricEmbedding feature map for each dimension of feature.
    
    Maps each feature dimension to TrigonometricEmbedding features.
    
    Attributes
    ----------
    k : int
        Number of frequency components per dimension
    input_shape : tuple
        Shape of input (n_features, n_dims_per_feature)
    """
    
    def __init__(self, k: int = 1, input_shape: tuple = (2, 2), **kwargs):
        """Initialize the TrigonometricEmbedding chain embedding.
        
        Parameters
        ----------
        k : int, optional
            Number of frequency components, by default 1
        input_shape : tuple, optional
            Input shape (n_features, n_dims_per_feature), by default (2, 2)
        **kwargs : Any
            Additional arguments passed to parent class
            
        Raises
        ------
        AssertionError
            If k < 1
        """
        assert k >= 1, "k must be at least 1"
        self.k = k
        self.input_shape = input_shape
        super().__init__(**kwargs)

    @property
    def dims(self) -> Collection[int]:
        """Get output dimensions for each feature."""
        return [self.k * 2 * self.input_shape[1]] * self.input_shape[0]

    @property
    def input_dims(self) -> jnp.ndarray:
        """Get input dimensions for each feature."""
        return jnp.array([1] * self.k)
    
    @property
    def embeddings(self) -> Collection[Embedding]:
        """Get TrigonometricEmbedding embeddings for each dimension."""
        return [TrigonometricEmbedding(k=self.k) for _ in range(self.input_shape[1])]

    def __call__(self, x: Collection) -> jnp.ndarray:
        """Apply TrigonometricEmbedding chain embedding to input.
        
        Parameters
        ----------
        x : Collection
            Input features
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Concatenated TrigonometricEmbedding features
        """
        embedded = []
        for f, xi in zip(self.embeddings, x):
            embedded.extend(f(xi))
        return jnp.array(embedded)
    
class TrigonometricEmbeddingAvg(ComplexEmbedding):
    """TrigonometricEmbedding feature map for mean of features.
    
    Maps the mean of input features to TrigonometricEmbedding features.
    
    Attributes
    ----------
    k : int
        Number of frequency components
    input_shape : tuple
        Shape of input (n_features, n_dims_per_feature)
    """
    
    def __init__(self, k: int = 1, input_shape: tuple = (2, 2), **kwargs):
        """Initialize the TrigonometricEmbedding average embedding.
        
        Parameters
        ----------
        k : int, optional
            Number of frequency components, by default 1
        input_shape : tuple, optional
            Input shape (n_features, n_dims_per_feature), by default (2, 2)
        **kwargs : Any
            Additional arguments passed to parent class
            
        Raises
        ------
        AssertionError
            If k < 1
        """
        assert k >= 1, "k must be at least 1"
        self.k = k
        self.input_shape = input_shape
        super().__init__(**kwargs)

    @property
    def dims(self) -> int:
        """Get output dimensions for each feature."""
        return [self.k * 2] * self.input_shape[0]

    @property
    def input_dims(self) -> jnp.ndarray:
        """Get input dimensions for each feature."""
        return jnp.array([1] * self.k)

    @property
    def embeddings(self) -> Embedding:
        """Get TrigonometricEmbedding embedding."""
        return TrigonometricEmbedding(k=self.k)

    def __call__(self, features: Collection) -> jnp.ndarray:
        """Apply TrigonometricEmbedding average embedding to input.
        
        Parameters
        ----------
        features : Collection
            Input features
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            TrigonometricEmbedding features of mean
        """
        return self.embeddings(jnp.mean(features))
    
class BasePatchEmbedding(StateVectorToMPSEmbedding):
    """Base class for patch-based embeddings that convert input data to MPS.
    
    Attributes
    ----------
    k : int
        Kernel size of patch window (k×k)
    mps : Optional[qtn.MatrixProductState]
        Current MPS representation
    """
    
    def __init__(self, k: int = 2, **kwargs):
        """Initialize the base patch embedding.
        
        Parameters
        ----------
        k : int, optional
            Kernel size of patch window, by default 2
        **kwargs : Any
            Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.k = k
        self.mps = None

    @property
    def dims(self) -> list:
        """Get dimensions of the MPS tensors."""
        return list([tensor.shape for tensor in self.mps.tensors])
    
    def pad_or_truncate_statevector(self, statevector: jnp.ndarray, target_size: int) -> jnp.ndarray:
        """Pad or truncate statevector to target size.
        
        Parameters
        ----------
        statevector : :class:`jax.numpy.ndarray`
            Input statevector
        target_size : int
            Target size
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Padded or truncated statevector
        """
        current_size = statevector.shape[0]

        if current_size < target_size:
            padding = [(0, target_size - current_size)]
            statevector = jnp.pad(statevector, padding, mode='constant')
        else:
            statevector = statevector[:target_size]
        
        return statevector

    def combine_mps_patches(self, mps_patches: onp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Combine MPS patches into single MPS.
        
        Parameters
        ----------
        mps_patches : onp.ndarray
            List of MPS patches
        n_qubits : int
            Number of qubits
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Combined MPS arrays
        """
        new_arrays = []
        number_interval = 0
        
        for patch in mps_patches:
            for i, arr in enumerate(patch):
                if i == number_interval * n_qubits and len(arr.shape) == 2:
                    new_arrays.append(jnp.expand_dims(arr, axis=0))
                elif i == ((number_interval + 1) * n_qubits - 1) and len(arr.shape) == 2:
                    new_arrays.append(jnp.expand_dims(arr, axis=-1))
                    number_interval += 1
                else:
                    new_arrays.append(arr)

        return new_arrays
    
    @property
    @abc.abstractmethod
    def create_statevector(self, x: jnp.ndarray) -> jnp.ndarray:
        """Create statevector from input data.
        
        Parameters
        ----------
        x : :class:`jax.numpy.ndarray`
            Input data
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Statevector representation
        """
        pass
    
    def __call__(self, x: jnp.ndarray) -> qtn.MatrixProductState:
        """Convert input data to MPS representation.
        
        Parameters
        ----------
        x : :class:`jax.numpy.ndarray`
            Input data (typically an image)
            
        Returns
        -------
        qtn.MatrixProductState
            MPS representation
            
        Raises
        ------
        ValueError
            If input is not square or patch size is too large
        """
        H, W = x.shape
        if H != W:
            raise ValueError(f"Only square matrices are supported, got {H}x{W} image.")
        if self.k > H:
            raise ValueError(f"Patch dimension k = {self.k} is too large for {H}x{W} images.")
        
        patches = u.divide_into_patches(x, self.k)
        mps_patches = []
        
        for patch in patches:
            patch_data = patch.ravel() if not hasattr(self, 'flatten_snake') else self.flatten_snake(patch)
            statevector, n_qubits = self.create_statevector(patch_data)
            mps_arrays = u.from_dense_to_mps(statevector, n_qubits, self.max_bond)
            mps_patches.append(mps_arrays)

        new_arrays = self.combine_mps_patches(mps_patches, n_qubits)
        
        # Recreate the MPS with the reshaped arrays
        self.mps = qtn.MatrixProductState(new_arrays, shape='lrp')
        return self.mps
    
class PatchEmbedding(BasePatchEmbedding):
    """Embedding that converts image patches to MPS using basis encoding."""
    
    def flatten_snake(self, image: jnp.ndarray) -> jnp.ndarray:
        """Flatten image in snake-like fashion.
        
        Parameters
        ----------
        image : :class:`jax.numpy.ndarray`
            Input image
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Flattened image in snake-like order
        """
        image = jnp.where(
            jnp.arange(image.shape[0])[:, None] % 2 == 1,
            jnp.flip(image, axis=1),
            image
        )
        return image.reshape(-1)

    def create_statevector(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
        """Create statevector using basis encoding.
        
        Parameters
        ----------
        x : :class:`jax.numpy.ndarray`
            Input patch data
            
        Returns
        -------
        Tuple[:class:`jax.numpy.ndarray`, int]
            Statevector and number of qubits
        """
        # Number of pixels (N = 16 for a 4x4 image)
        N = len(x)
        # Number of address qubits is log2(N) = 4
        n_address_qubits = int(onp.ceil(onp.log2(N)))
        # One color qubit
        n_color_qubit = 1
        # Total number of qubits = address qubits + 1 color qubit
        n_qubits = n_address_qubits + n_color_qubit
        
        # Create index tensors for addressing
        state_indices = jnp.arange(2**n_qubits)
        # Color qubit is the least significant bit
        color_bits = state_indices % 2
        # Address qubits are the most significant bits
        address_indices = state_indices // 2

        # Calculate cos and sin for each pixel intensity
        cos_values = jnp.cos(math.pi * x / 2)
        sin_values = jnp.sin(math.pi * x / 2)
        
        # Create the statevector with color qubit encoding
        statevector = jnp.where(
            color_bits == 0,
            cos_values[address_indices],
            sin_values[address_indices]
        )

        # Normalize the statevector
        statevector /= jnp.linalg.norm(statevector)

        # Pad or truncate to fixed size
        fixed_size = 2**n_qubits
        padded_statevector = self.pad_or_truncate_statevector(statevector.flatten(), fixed_size)

        return padded_statevector, n_qubits

class PatchAmplitudeEmbedding(BasePatchEmbedding):
    """Embedding that converts image patches to MPS using amplitude encoding."""
    
    def create_statevector(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
        """Create statevector using amplitude encoding.
        
        Parameters
        ----------
        x : :class:`jax.numpy.ndarray`
            Input patch data
            
        Returns
        -------
        Tuple[:class:`jax.numpy.ndarray`, int]
            Statevector and number of qubits
        """
        N = len(x)
        n_qubits = int(onp.ceil(onp.log2(N)))
        
        statevector = jnp.sqrt(x)
        statevector /= jnp.linalg.norm(statevector)
        
        fixed_size = 2**n_qubits
        padded_statevector = self.pad_or_truncate_statevector(statevector.flatten(), fixed_size)
        
        return padded_statevector, n_qubits


def embed(x: onp.ndarray, phi: Union[Embedding, ComplexEmbedding, StateVectorToMPSEmbedding], **mps_opts) -> qtn.MatrixProductState:
    """Create product state from feature vector.
    
    Works only if features are separated and not correlated.
    
    Parameters
    ----------
    x : onp.ndarray
        Vector or matrix of features
    phi : Union[Embedding, ComplexEmbedding, StateVectorToMPSEmbedding]
        Feature map for each feature
    **mps_opts : Any
        Additional arguments passed to MatrixProductState
        
    Returns
    -------
    qtn.MatrixProductState
        Product state representation
        
    Raises
    ------
    TypeError
        If phi is not a valid embedding type
    """
    if not issubclass(type(phi), (ComplexEmbedding, Embedding, StateVectorToMPSEmbedding)):
        raise TypeError('Invalid embedding type')
    
    if issubclass(type(phi), Embedding):
        arrays = [phi(xi).reshape((1, 1, phi.dim)) for xi in x]
        for i in [0, -1]:
            arrays[i] = arrays[i].reshape((1, phi.dim))
        mps = qtn.MatrixProductState(arrays, **mps_opts)
    
    elif issubclass(type(phi), ComplexEmbedding) and x.ndim == 2:
        if type(phi.dims) == int:
            arrays = [phi(xi).reshape((1, 1, phi.dims)) for xi in x]
            for i in [0, -1]:
                arrays[i] = arrays[i].reshape((1, phi.dims))
        else:
            arrays = [phi(xi).reshape((1, 1, phi.dims[i])) for i, xi in enumerate(x)]
            for i in [0, -1]:
                arrays[i] = arrays[i].reshape((1, phi.dims[i]))

        mps = qtn.MatrixProductState(arrays, **mps_opts)
    else:
        mps = phi(x)
    
    # Normalize
    if len(mps.tensors) > 200:  # For large systems
        for i, tensor in enumerate(mps.tensors):
            if i == 0:
                mps.left_canonize_site(i)
            elif i == len(mps.tensors) - 1:
                tensor.modify(data=tensor.data / jnp.linalg.norm(tensor.data))
            else:
                tensor.modify(data=tensor.data / jnp.linalg.norm(tensor.data))
                mps.left_canonize_site(i)
    else:
        norm = mps.norm()
        for tensor in mps.tensors:
            tensor.modify(data=tensor.data / jnp.power(norm, 1 / len(mps.tensors)))

    return mps
