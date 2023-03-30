import abc
import itertools
from numbers import Number

import numpy as onp
from autoray import numpy as np
import quimb.tensor as qtn
import quimb as qu
import jax
import math


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
        """Mapping dimension.
        """
        pass

    @abc.abstractmethod
    def __call__(self, x: Number) -> onp.ndarray:
        pass


class trigonometric(Embedding):
    """Trigonometric feature map.
    """
    def __init__(self, k: int = 1, **kwargs):
        """Constructor

        Attributes
        ----------
        k : int
            Custom parameter = ``dim/2``.
        """
        assert k >= 1

        self.k = 1
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        return self.k * 2

    def __call__(self, x: Number) -> onp.ndarray:
        return 1 / np.sqrt(self.k) * np.asarray([f(onp.pi * x / 2**i) for f, i in itertools.product([np.cos, np.sin], range(1, self.k + 1))])


class fourier(Embedding):
    """Fourier feature map.
    """
    def __init__(self, p: int = 2, **kwargs):
        """Constructor

        Attributes
        ----------
        p : int
            Mapping dimension.
        """
        assert p >= 2

        self.p = p
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        return self.p

    def __call__(self, x: Number) -> onp.ndarray:
        return 1 / self.p * np.asarray([np.abs(sum((np.exp(1j * 2 * onp.pi * k * ((self.p - 1) * x - j) / self.p) for k in range(self.p)))) for j in range(self.p)])
    

class linear(Embedding):
    """Linear feature map.
        [x, 1-x] where x = feature in range [0,1]
    """
    def __init__(self, **kwargs):
        """Constructor

        """        
        self.p = 2
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        return self.p

    def __call__(self, x: Number) -> onp.ndarray:
        return np.asarray([x, 1-x])

def physics_embedding(data: onp.ndarray, pT_embed_func: Embedding, **mps_opts):
    theta = data[:, 0]
    phi = data[:, 1]
    pT = data[:, 2]

    # encode pT
    pT_embed = [pT_embed_func(pt) for pt in pT]

    # add phi and theta
    data_embed = []
    for i, j, k in zip(phi, theta, pT_embed):
        pt_norm = np.linalg.norm(k)
        pt_phi_norm = np.linalg.norm(np.asarray([k[0]*np.cos(i), k[1]*np.cos(i), pt_norm*np.sin(i)]))
        data_vector = np.asarray([k[0]*np.cos(i)*np.cos(j), k[1]*np.cos(i)*np.cos(j), pt_norm*np.sin(i)*np.cos(j), pt_phi_norm*np.sin(j)])
        data_embed.append(data_vector)

    # reshape for mps
    data_embed_reshaped = [x.reshape((1,1,4)) for x in data_embed]

    return qtn.MatrixProductState(data_embed_reshaped, **mps_opts)

def embed(x: onp.ndarray, phi: Embedding, **mps_opts):
    """Creates a product state from a vector of features `x`.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Vector of features.
    phi: :class:`tn4ml.embeddings.Embedding`
        Embedding type.
    mps_opts: optional
        Additional arguments passed to MatrixProductState class.
    """
    if x.ndim > 1:
        jets = True
    else: jets = False

    if jets:
        return physics_embedding(x, phi, **mps_opts)
    else:
        arrays = [phi(xi).reshape((1, 1, phi.dim)) for xi in x]
        for i in [0, -1]:
            arrays[i] = arrays[i].reshape((1, phi.dim))

        return qtn.MatrixProductState(arrays, **mps_opts)