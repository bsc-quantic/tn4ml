import abc
import itertools
import math
from numbers import Number
from typing import Collection, Any, Union

import numpy as onp
import jax.numpy as jnp
from jax import lax
import quimb.tensor as qtn

import tn4ml.util as u
from tn4ml.scipy.special import eval_legendre, eval_laguerre, eval_hermite

class Embedding:
    """Data embedding (feature map) class.

    Attributes
    ----------
        dtype: :class:`numpy.dtype`
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
        """Embedding function."""
        pass

class ComplexEmbedding:
    """Complex data embedding (feature map) class where each feature has its own choosen embedding.

    Attributes
    ----------
        dtype: :class:`numpy.dtype`
            Data Type
    """
    def __init__(self, dtype=onp.float32):
        self.dtype = dtype

    @property
    @abc.abstractmethod
    def dims(self) -> Collection[int]:
        """ Mapping dimensions per feature """
        pass

    @property
    @abc.abstractmethod
    def input_dims(self) -> jnp.ndarray:
        """ Dimensionality of each input feature. 1 = number, 2 = vector """
        pass

    @property
    @abc.abstractmethod
    def embeddings(self) -> Collection[Embedding]:
        """ Embedding for each feature """
        pass

    @abc.abstractmethod
    def __call__(self, x: Number) -> jnp.ndarray:
        pass

class StateVectorToMPSEmbedding:
    """
    A class to convert a statevector into a Matrix Product State (MPS).
    """
    def __init__(self, dtype=onp.float32, max_bond=None):
        self.dtype = dtype
        self.max_bond = max_bond

    @property
    @abc.abstractmethod
    def dims(self) -> list:
        """ Dimensions of mps arrays """
        pass

    @property
    @abc.abstractmethod
    def create_statevector(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Method to create a statevector """
        pass
    
    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Method to convert a Statevector into an Matrix Product State """
        pass

class MPSEmbedding:
    """
    An abstract class to convert a input image into a Matrix Product State (MPS) with your custom decomposition strategy.
    """
    def __init__(self, dtype=onp.float32, max_bond=None):
        self.dtype = dtype
        self.max_bond = max_bond

    @property
    @abc.abstractmethod
    def dims(self) -> list:
        """ Dimensions of mps arrays """
        pass

    @property
    @abc.abstractmethod
    def decompose(self, x: Any, *args) -> jnp.ndarray:
        """ Method to decompose an """
        pass
    
    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Method to convert an input into an Matrix Product State """
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
    
    Computes cosine and sine components for different frequencies:
    :math:`\\phi(x) = \\frac{1}{\\sqrt{p}}[\\cos(2\\pi 0 x), \\cos(2\\pi 1 x), ..., \\cos(2\\pi (p-1) x), \\sin(2\\pi 0 x), \\sin(2\\pi 1 x), ..., \\sin(2\\pi (p-1) x)]`
    
    Attributes
    ----------
    p: int
        Number of frequency components.
    """

    def __init__(self, p: int = 2, **kwargs):
        assert p >= 1, "Number of frequency components must be at least 1"
        self.p = p
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """ Mapping dimension """
        return 2 * self.p  # p cosine terms + p sine terms

    @property
    def input_dim(self) -> int:
        return 1

    def __call__(self, x: Number) -> jnp.ndarray:
        """Embedding function for Fourier features.
        
        Parameters
        ----------
        x: :class:`Number`
            Input feature in [0,1].
        
        Returns
        -------
        jnp.ndarray
            Embedding vector with cosine and sine components.
        """
        k = jnp.arange(self.p)  # Frequency indices
        features = jnp.concatenate([jnp.cos(2 * jnp.pi * k * x), jnp.sin(2 * jnp.pi * k * x)])
        return features / jnp.sqrt(self.p)  # Normalization for orthonormality
    
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
        x: :class:`Number`
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
        return jnp.prod(self.centers.shape)
    
    @property
    def input_dim(self) -> int:
        """Dimensionality of input feature. 1 = number"""
        return 1
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embedding function for Gaussian RBF.
        
        Parameters
        ----------
        x : :class:`Number`
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
    n : int
        Number of features.
    include_bias : bool
        Include bias term.
    """

    def __init__(self, degree: int, n: int, include_bias: bool = False, **kwargs):
        if degree < 1:
            raise ValueError("Degree of polynomial embedding must be at least 1.")
        self.degree = degree
        self.n = n
        self.include_bias = include_bias
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """ Mapping dimension """
        if self.include_bias:
            return sum(math.comb(self.input_dim + k - 1, k) for k in range(0, self.degree + 1))
        else:
            return sum(math.comb(self.input_dim + k - 1, k) for k in range(1, self.degree + 1))

    @property
    def input_dim(self) -> int:
        """ Dimensionality of input feature"""
        return self.n
    
    def __call__(self, x: Union[Number, onp.array]) -> jnp.ndarray:
        """Embedding function for polynomial.
        
        Parameters
        ----------
        x : :class:`Number`
            Input feature.
        
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """

        if x.ndim == 0:
            x = jnp.array([x])
        
        features = []
        if self.include_bias:
            features.append(1.0)        
        
        # Generate combinations of feature indices with repetition up to the specified degree
        for d in range(1, self.degree + 1):
            for combination in itertools.combinations_with_replacement(range(len(x)), d):
                # Compute the product of the selected features
                product = jnp.prod(x[jnp.array(combination)])
                features.append(product)
        
        return jnp.array(features)

class legendre(Embedding):
    """Legendre polynomial feature map.
    
    Attributes
    ----------
    degree : int
        Maximum polynomial degree (inclusive)
    """
    
    def __init__(self, degree: int = 2, **kwargs):
        self.degree = degree
        super().__init__(**kwargs)
        
    @property
    def dim(self) -> int:
        """Mapping dimension"""
        return self.degree + 1
        
    @property
    def input_dim(self) -> int:
        return 1
        
    def __call__(self, x: Number) -> jnp.ndarray:
        """Embedding function for Legendre polynomials.
        
        Parameters
        ----------
        x: :class:`Number`
            Input feature in [-1, 1].
            
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """
        features = jnp.array([eval_legendre(k, x) for k in range(self.degree + 1)])
        return features
    
class laguerre(Embedding):
    """Laguerre polynomial feature map with isometric weighting.
    
    Computes :math:`e^(-x/2) * L_k(x)` for k from 0 to degree.
    
    Attributes
    ----------
    degree : int
        Maximum polynomial degree (inclusive)
    """
    
    def __init__(self, degree: int = 2, **kwargs):
        self.degree = degree
        super().__init__(**kwargs)
        
    @property
    def dim(self) -> int:
        """Mapping dimension"""
        return self.degree + 1
        
    @property
    def input_dim(self) -> int:
        return 1
        
    def __call__(self, x: Number) -> jnp.ndarray:
        """Embedding function for weighted Laguerre polynomials.
        
        Parameters
        ----------
        x: :class:`Number`
            Input feature in [0, ∞).
            
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """
        weight = jnp.exp(-x / 2)
        features = jnp.array([weight * eval_laguerre(k, x) for k in range(self.degree + 1)])
        return features

class hermite(Embedding):
    """Hermite polynomial feature map with isometric weighting.
    
    Computes :math:`e^(-x^2/2) * H_k(x)` for k from 0 to degree.
    
    Attributes
    ----------
    degree : int
        Maximum polynomial degree (inclusive)
    """
    
    def __init__(self, degree: int = 2, **kwargs):
        self.degree = degree
        super().__init__(**kwargs)
        
    @property
    def dim(self) -> int:
        """Mapping dimension"""
        return self.degree + 1
        
    @property
    def input_dim(self) -> int:
        return 1
        
    def __call__(self, x: Number) -> jnp.ndarray:
        """Embedding function for weighted Hermite polynomials.
        
        Parameters
        ----------
        x: :class:`Number`
            Input feature in R (real numbers).
            
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """
        weight = jnp.exp(-0.5 * x**2)
        features = jnp.array([weight * eval_hermite(k, x) for k in range(self.degree + 1)])
        return features
 
class jax_arrays(Embedding):
    """Input arrays to JAX arrays. No embedding.
    Optional: adding one to the input array.
    
    Attributes
    ----------
    add_bias: bool
        Add bias term (1.0)
    """
    def __init__(self, dim: int = None, add_bias: bool = False, input_dim: int = None, **kwargs):
        super().__init__(**kwargs)
        self._dim = dim
        self.add_bias = add_bias
        self._input_dim = input_dim

    @property
    def dim(self) -> int:
        """ Mapping dimension """
        return self._dim
    
    @property
    def input_dim(self) -> int:
        return self._input_dim
    
    def __call__(self, x: Any) -> jnp.ndarray:
        """Embedding function for JAX arrays.
        
        Parameters
        ----------
        x: list
            List of input features.
        
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """
        if self.add_bias:
            return jnp.concatenate([jnp.array([1.]), x])
        return jnp.array(x)
    
class trigonometric_chain(ComplexEmbedding):
    """ Trigonometric feature map for each dimension of feature.
    Sample = [[x11, x12, x13], [x21, x22, x23]] ==> [[cos(x11), sin(x11), cos(x12), sin(x12), cos(x13), sin(x13)], [cos(x21), sin(x21), cos(x22), sin(x22), cos(x23), sin(x23)]]
        - dims = 6 for each feature.
        - input_dims = 1 for each feature.

    Attributes
    ----------
    k: int
        Custom parameter = ``dim/2``.
    
    input_shape: tuple
        Input shape: number of features and number of dimensions per feature.
    """

    def __init__(self, k: int = 1, input_shape: tuple = (2, 2), **kwargs):
        assert k >= 1
        self.k = k
        self.input_shape = input_shape
        super().__init__(**kwargs)

    @property
    def dims(self) -> Collection[int]:
        """ Mapping dimensions per feature """
        return [self.k * 2 * self.input_shape[1]]*self.input_shape[0]

    @property
    def input_dims(self) -> jnp.ndarray:
        return jnp.array([1] * self.k)
    
    @property
    def embeddings(self) -> Collection[Embedding]:
        return [trigonometric(k=self.k) for _ in range(self.input_shape[1])]

    def __call__(self, x: Collection) -> jnp.ndarray:
        """Embedding function for trigonometric chain.
        
        Parameters
        ----------
        x: list
            List of input features.
        
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """
        embedded = []
        for f, xi in zip(self.embeddings, x):
            embedded.extend(f(xi))
        return jnp.array(embedded)
    
class trigonometric_avg(ComplexEmbedding):
    """ Trigonometric feature map for mean(features).

    Attributes
    ----------
    k: int
        Custom parameter = ``dim/2``.
    """

    def __init__(self, k: int = 1, input_shape: tuple = (2, 2), **kwargs):
        assert k >= 1
        self.k = k
        self.input_shape = input_shape
        super().__init__(**kwargs)

    @property
    def dims(self) -> int:
        """ Mapping dimensions per feature """
        return [self.k * 2]*self.input_shape[0]

    @property
    def input_dims(self) -> jnp.ndarray:
        return jnp.array([1] * self.k)

    @property
    def embeddings(self) -> Embedding:
        return trigonometric(k=self.k)

    def __call__(self, features: Collection) -> jnp.ndarray:
        """
        Embedding function for average of input features.

        Parameters
        ----------
        features: list
            List of input features.

        Returns
        -------
        :class:`jax.numpy.ndarray`
            Embedded vector.
        """
        return self.embeddings(jnp.mean(features))
    
class BasePatchEmbedding(StateVectorToMPSEmbedding):
    """
    Base class for patch-based embedding methods that convert input data to MPS.
    
    Attributes
    ----------
    k : int
        The kernel size of the patch window (k×k).
    """
    def __init__(self, k=2, **kwargs):
        """
        Initialize the BasePatchEmbedding class.

        Parameters
        ----------
        k: int
            The kernel size of the patch window kxk.
        """
        super().__init__(**kwargs)
        self.k = k
        self.mps = None

    @property
    def dims(self) -> list:
        """Get dimensions of the MPS tensors"""
        return list([tensor.shape for tensor in self.mps.tensors])
    
    def pad_or_truncate_statevector(self, statevector: jnp.ndarray, target_size: int) -> jnp.ndarray:
        """
        Pad or truncate the statevector to a target size.

        Parameters
        ----------
        statevector: :class:`jax.numpy.ndarray`
            The input statevector.
        target_size: int
            The desired size of the statevector.

        Returns
        -------
        :class:`jax.numpy.ndarray`
            A statevector of the target size.
        """
        current_size = statevector.shape[0]

        # Pad or truncate
        if current_size < target_size:
            # Pad with zeros if smaller than target size
            padding = [(0, target_size - current_size)]
            statevector = jnp.pad(statevector, padding, mode='constant')
        else:
            # Truncate if larger than target size
            statevector = statevector[:target_size]
        
        return statevector

    def combine_mps_patches(self, mps_patches: onp.ndarray, n_qubits: int) -> jnp.ndarray:
        """
        Combine arrays of each MPS patch into a single MPS.

        Parameters
        ----------
        mps_patches: :class:`numpy.ndarray`
            List of MPS patches (nested lists of arrays).
        n_qubits: int
            Number of qubits.

        Returns
        -------
        :class:`jax.numpy.ndarray`
            A list of arrays for combined MPS.
        """
        new_arrays = []
        number_interval = 0
        
        for patch in mps_patches:
            for i, arr in enumerate(patch):
                # Check if current array index matches the start or end of an interval
                if i == number_interval * n_qubits and len(arr.shape) == 2:
                    # Add a new axis at the beginning (dim=0)
                    new_arrays.append(jnp.expand_dims(arr, axis=0))
                elif i == ((number_interval + 1) * n_qubits - 1) and len(arr.shape) == 2:
                    # Add a new axis at the end (dim=-1)
                    new_arrays.append(jnp.expand_dims(arr, axis=-1))
                    number_interval += 1
                else:
                    # Add the array as is
                    new_arrays.append(arr)

        return new_arrays
    
    @property
    @abc.abstractmethod
    def create_statevector(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Abstract method to create a statevector representation of an input array.
        Must be implemented by subclasses.
        """
        pass
    
    def __call__(self, x: jnp.ndarray) -> qtn.MatrixProductState:
        """
        Convert input data into a Matrix Product State (MPS).
        
        Parameters
        ----------
        x: :class:`jax.numpy.ndarray`
            Input data (typically an image).
        
        Returns
        -------
        :class:`quimb.tensor.MatrixProductState`
            A Matrix Product State representation of the input data.
        """
        H, W = x.shape  # H: height, W: width
        if H != W:
            raise ValueError(f"Only square matrices are supported, got {H}x{W} image.")
        if self.k > H:
            raise ValueError(f"Patch dimension k = {self.k} is too large for {H}x{W} images.")
        
        patches = u.divide_into_patches(x, self.k)
        mps_patches = []
        
        # Process each patch
        for patch in patches:
            # Subclasses will implement different statevector creation methods
            patch_data = patch.ravel() if not hasattr(self, 'flatten_snake') else self.flatten_snake(patch)
            statevector, n_qubits = self.create_statevector(patch_data)
            mps_arrays = u.from_dense_to_mps(statevector, n_qubits, self.max_bond)
            mps_patches.append(mps_arrays)

        new_arrays = self.combine_mps_patches(mps_patches, n_qubits)
        
        # Recreate the MPS with the reshaped arrays
        self.mps = qtn.MatrixProductState(new_arrays, shape='lrp')
        return self.mps
    
class PatchEmbedding(BasePatchEmbedding):
    """
    Embedding that converts image patches to MPS using basis encoding.
    """
    def flatten_snake(self, image: jnp.ndarray) -> jnp.ndarray:
        """
        Flatten an image in a snake-like fashion.
        
        Parameters
        ----------
        image: :class:`jax.numpy.ndarray`
            A 2D array of pixel intensities.
            
        Returns
        -------
        :class:`jax.numpy.ndarray`
            A 1D array of pixel intensities in snake-like order.
        """
        # Flip every other row by slicing
        image = jnp.where(jnp.arange(image.shape[0])[:, None] % 2 == 1, jnp.flip(image, axis=1), image)
        
        # Flatten the image
        flattened = image.reshape(-1)
        
        return flattened

    def create_statevector(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Create a statevector representation using basis encoding.

        Parameters
        ----------
        x: :class:`jax.numpy.ndarray`
            An array of patch pixel intensities.

        Returns
        -------
        tuple
            A tuple containing the statevector and number of qubits.
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
        color_bits = state_indices % 2  # Extract color qubit (last bit)
        address_indices = state_indices // 2  # Extract address state

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
    """
    Embedding that converts image patches to MPS using amplitude encoding.
    """
    def create_statevector(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Create a statevector representation using amplitude encoding.

        Parameters
        ----------
        x: :class:`jax.numpy.ndarray`
            An array of patch pixel intensities.

        Returns
        -------
        tuple
            A tuple containing the statevector and number of qubits.
        """
        # Number of pixels
        N = len(x)
        # Number of qubits needed
        n_qubits = int(onp.ceil(onp.log2(N)))
        
        # Create the state vector and fill it with square roots of the pixel values
        statevector = jnp.sqrt(x)

        # Normalize the statevector
        statevector /= jnp.linalg.norm(statevector)
        
        # Pad or truncate to fixed size
        fixed_size = 2**n_qubits
        padded_statevector = self.pad_or_truncate_statevector(statevector.flatten(), fixed_size)
        
        return padded_statevector, n_qubits


def embed(x: onp.ndarray, phi: Union[Embedding, ComplexEmbedding, StateVectorToMPSEmbedding], **mps_opts):
    """
    Creates a product state from a vector of features `x`.
    Works only if features are separated and not correlated (this check you need to do yourself).

    Parameters
    ----------
    x: :class:`numpy.ndarray`
        Vector or Matrix of features.
    phi: :class:`tn4ml.embeddings.Embedding` or :class:`tn4ml.embeddings.ComplexEmbedding` or :class:`tn4ml.embeddings.StateVectorToMPSEmbedding`
        Feature map for each feature.
    mps_opts: Optional parameters.
        Additional arguments passed to MatrixProductState class.
    """
    if not issubclass(type(phi), ComplexEmbedding) and not issubclass(type(phi), Embedding) and not issubclass(type(phi), StateVectorToMPSEmbedding):
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
    
    # normalize
    if len(mps.tensors) > 200: # for large systems
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
