import abc
import itertools
import math
from numbers import Number
from typing import Collection, Any, Union
import numpy as onp
from autoray import numpy as np
import autoray as a
import jax.numpy as jnp
from jax import lax
import jax
import quimb.tensor as qtn
import tn4ml.util as u

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

class ComplexEmbedding:
    """Complex data embedding (feature map) class where each feature has its own choosen embedding.

    Attributes
    ----------
        dype: :class:`numpy.dype`
            Data Type
    """
    def __init__(self, dtype=onp.float32):
        self.dtype = dtype

    @property
    @abc.abstractmethod
    def dims(self) -> int:
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
        """ Embeddings for each feature """
        pass

    @abc.abstractmethod
    def __call__(self, x: Number) -> jnp.ndarray:
        pass

class StateVectorToMPSEmbedding:
    """
    A class to convert a statevector into a Matrix Product State (MPS).
    """
    def __init__(self, dtype=onp.float32, max_bond=None):
        self.dype = dtype
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
    
class trigonometric_chain(ComplexEmbedding):
    """ Trigonometric feature map for each feature.

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
    def dims(self) -> int:
        """ Mapping dimensions per feature """
        return [self.k * 2] * self.k

    @property
    def input_dims(self) -> jnp.ndarray:
        return jnp.array([1] * self.k)
    
    @property
    def embeddings(self) -> Collection[Embedding]:
        return [trigonometric(k=self.k) for _ in range(self.k)]

    def __call__(self, x: Number) -> jnp.ndarray:
        """Embedding function for trigonometric chain.
        
        Parameters
        ----------
        x: :class:`Number`
            Input feature.
        
        Returns
        -------
        jnp.ndarray
            Embedding vector.
        """
        return jnp.concatenate([f(x) for f in self.embeddings])
    
class PatchEmbedding(StateVectorToMPSEmbedding):
    def __init__(self, k = 2, **kwargs):
        """
        Initialize the PatchedEmbedding class.

        Parameters
        ----------
        k: int
            The kernel size of the patch window kxk.
        
        Returns
        -------
        None
        """
        super().__init__(**kwargs)
        self.k = k
        self.mps = None

    @property
    def dims(self) -> list:
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

    def create_statevector(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Create a statevector representation of an input array (vector like).

        Parameters
        ----------
        x: :class:`jax.numpy.ndarray`
            An array of patch pixel intensities flattened from original patch k x k.

        Returns
        -------
        :class:`jax.numpy.ndarray`
            A statevector representation of the input array.
        """
         # Number of pixels (N = 16 for a 4x4 image)
        N = len(x)
        # Number of address qubits is log2(N) = 4
        n_address_qubits = int(np.log2(N))
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

    def __call__(self, x: jnp.ndarray) -> qtn.MatrixProductState:
        """
        Convert a Statevector into a Matrix Product State (MPS).

        Parameters
        ----------
        x: :class:`jax.numpy.ndarray`
            A Statevector.
        
        Returns
        -------
        :class:`quimb.tensor.MatrixProductState`
            A Matrix Product State representation of the input Statevector.
        """

        H, W = x.shape  # H: height, W: width patches: number of patches
        if H != W:
            raise ValueError("Only square matrix input is supported.")
        
        patches = u.divide_into_patches(x, self.k)
        mps_patches = []
        for patch in patches:
            patch_pixels = self.flatten_snake(patch)
            statevector, n_qubits = self.create_statevector(patch_pixels)
            mps_arrays = u.from_dense_to_mps(statevector, n_qubits, self.max_bond)

            mps_patches.append(mps_arrays)

        new_arrays = self.combine_mps_patches(mps_patches, n_qubits)
        
        # Recreate the MPS with the reshaped arrays
        self.mps = qtn.MatrixProductState(new_arrays, shape='lrp')
        return self.mps

def embed(x: onp.ndarray, phi: Union[Embedding, ComplexEmbedding, StateVectorToMPSEmbedding], **mps_opts):
    """Creates a product state from a vector of features `x`.
    Works only if features are separated and not correlated (this check you need to do yourself).

    Parameters
    ----------
    x: :class:`numpy.ndarray`
        Vector or Matrix of features.
    phi: :class:`tn4ml.embeddings.Embedding`
        Feature map for each feature.
    mps_opts: optional
        Additional arguments passed to MatrixProductState class.
    """

    if not issubclass(type(phi), ComplexEmbedding) and not issubclass(type(phi), Embedding) and not issubclass(type(phi), StateVectorToMPSEmbedding):
        raise TypeError('Invalid embedding type')
    
    if issubclass(type(phi), Embedding):
        arrays = [phi(xi).reshape((1, 1, phi.dim)) for xi in x]
        for i in [0, -1]:
            arrays[i] = arrays[i].reshape((1, phi.dim))
        mps = qtn.MatrixProductState(arrays, **mps_opts)
    elif issubclass(type(phi), ComplexEmbedding):
        arrays = [phi.embeddings[i](xi).reshape((1, 1, phi.dims[i])) for i, xi in enumerate(x)]
        for i in [0, -1]:
            arrays[i] = arrays[i].reshape((1, phi.dims[i]))
        mps = qtn.MatrixProductState(arrays, **mps_opts)
    else:
        mps = phi(x)
    
    # normalize
    norm = mps.norm()
    for tensor in mps.tensors:
        tensor.modify(data=tensor.data / a.do("power", norm, 1 / len(mps.tensors)))
    return mps