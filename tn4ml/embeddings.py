import abc
import itertools
from numbers import Number
from typing import Union
import numpy as onp
from autoray import numpy as np
import quimb.tensor as qtn
import torch

class Embedding:
    """Data embedding (feature map) class.

    Attributes
    ----------
        dype: :class:`numpy.dype`
            Data Type
    """
    def __init__(self, dtype=onp.float64):
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

        self.k = k
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        return self.k * 2

    def __call__(self, x: Union[torch.Tensor, Number]) -> torch.Tensor:
        """Compute trigonometric feature map for input x.

        Parameters
        ----------
        x : Union[torch.Tensor, Number]
            Input value(s) for computing the feature map.

        Returns
        -------
        torch.Tensor
            Trigonometric feature map.
        """
        if isinstance(x, Number):
            x = torch.tensor(x)
        return 1 / torch.sqrt(torch.tensor(self.k)) * torch.stack([
            f((onp.pi * x / 2**i)) for f, i in itertools.product([torch.cos, torch.sin], range(1, self.k + 1))
        ]).view(-1)


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

def physics_embedding(data: onp.ndarray, embed_func: Embedding, **mps_opts):
    eta = data[:, 0]
    phi = data[:, 1]
    pT = data[:, 2]

    # encode pT
    pT_embed = np.asarray([embed_func(pt) for pt in pT])

    # add phi and theta
    data_embed = []
    for i, j, k in zip(phi, eta, pT_embed):
        pt_norm = np.linalg.norm(k)
        pt_phi_norm = np.linalg.norm(np.asarray([k[0]*np.cos(i), k[1]*np.cos(i), pt_norm*np.sin(i)]))
        data_vector = np.asarray([k[0]*np.cos(i)*np.cos(j), k[1]*np.cos(i)*np.cos(j), pt_norm*np.sin(i)*np.cos(j), pt_phi_norm*np.sin(j)])
        data_embed.append(data_vector)

    # reshape for mps
    data_embed_reshaped = [(x/np.linalg.norm(x)).reshape((1,1,4)) for x in data_embed]

    return qtn.MatrixProductState(data_embed_reshaped, **mps_opts)

def physics_embedding_5vector(data: onp.ndarray, embed_func: Embedding, **mps_opts):
    eta = data[:, 0]
    phi = data[:, 1]
    pT = data[:, 2]

    data_embed = []
    for e, p, pt in zip(eta, phi, pT):
        data_embed.append([pt, np.cos((onp.pi/2)*e), np.sin((onp.pi/2)*e), np.cos((onp.pi/2)*p), np.sin((onp.pi/2)*p)])
    
    data_embed = np.array(data_embed)
    # reshape for mps
    data_embed_reshaped = [(x/np.linalg.norm(x)).reshape((1,1,5)) for x in data_embed]

    return qtn.MatrixProductState(data_embed_reshaped, **mps_opts)

def physics_embedding_trig_pT(data: onp.ndarray, embed_func: Embedding, **mps_opts):
    deltaR = data[:, 0]
    pT = data[:, 1]

    # encode pT
    pT_embed = np.asarray([embed_func(pt) for pt in pT])

    # add phi and theta
    data_embed = []
    for i, k in zip(deltaR, pT_embed):
        pt_norm = np.linalg.norm(k)
        data_vector = np.asarray([k[0]*np.cos(i), k[1]*np.cos(i), pt_norm*np.sin(i)])
        data_embed.append(data_vector)
    
    data_embed_reshaped = [(x/np.linalg.norm(x)).reshape((1,1,3)) for x in data_embed]
    return qtn.MatrixProductState(data_embed_reshaped, **mps_opts)

def physics_embedding_pT_trig_deltaR(data: onp.ndarray, embed_func: Embedding, **mps_opts):
    deltaR = data[:, 0]
    pT = data[:, 1]

    data_embed = []
    for i, j in zip(deltaR, pT):
        data_vector = np.asarray([j, np.cos((onp.pi/2)*i), np.sin((onp.pi/2)*i)])
        data_embed.append(data_vector)
    
    data_embed_reshaped = [(x/np.linalg.norm(x)).reshape((1,1,3)) for x in data_embed]
    return qtn.MatrixProductState(data_embed_reshaped, **mps_opts)

def physics_embedding_cos_sin(data: onp.ndarray, embed_func: Embedding, **mps_opts):
    deltaR = data[:, 0]
    pT = data[:, 1]

    data_embed = []
    for i, j in zip(deltaR, pT):
        data_vector = np.asarray([np.cos(i), np.sin(j)])
        data_embed.append(data_vector)
    
    data_embed_reshaped = [(x/np.linalg.norm(x)).reshape((1,1,2)) for x in data_embed]
    return qtn.MatrixProductState(data_embed_reshaped, **mps_opts)

class amplitude_encoding_quatro(Embedding):
    def __init__(self, **kwargs):
        """Constructor

        """        
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        return 4

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x/torch.norm(x)
    
class whatever_encoding(Embedding):
    def __init__(self, dim, **kwargs):
        """Constructor

        """        
        super().__init__(**kwargs)
        self._dim = dim
        # self._std = std
        # self._mean = mean

    @property
    def dim(self) -> int:
        return self._dim

    # @property
    # def std(self) -> int:
    #     return self._std
    
    # @property
    # def mean(self) -> int:
    #     return self._mean

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x = (x+1)/2
        # std = torch.std(x)
        # mean = torch.mean(x)
        # # Avoid division by zero
        # std = torch.clamp(std, min=1e-6)

        # Perform zero-centering and normalization
        x = (x+1)/2
        return x

class original_inverse(Embedding):
    """Feature map `[x, 1-x]` or `[1, x, 1-x]`
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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
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
            vector = torch.zeros(2)
            vector[0] = x
            vector[1] = 1.0 - x
        elif self.p == 3:
            vector = torch.zeros(3)
            vector[0] = 1.0
            vector[1] = x
            vector[2] = 1.0 - x
        else:
            raise ValueError('Invalid dimension')
        return vector

def embed(x: onp.ndarray, phi: Embedding, phi_multidim: Embedding = None, **mps_opts):
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
    # if x.ndim > 1:
    #     multi_dim = False # TODO fix this!

    #     if not multi_dim:
    #         raise ValueError('Provide embedding function for multi-dimensional data.')
    # else:
    #     multi_dim = False
    arrays = [phi(xi).reshape((1, 1, phi.dim)) for xi in x]
    for i in [0, -1]:
        arrays[i] = arrays[i].reshape((1, phi.dim))
    # if pytorch:
    #     torch_arrays = [torch.tensor(np.float64(data)) for data in arrays]
    #     return qtn.MatrixProductState(torch_arrays, **mps_opts)
    # else:
    mps = qtn.MatrixProductState(arrays, **mps_opts)
    #mps.normalize()
    return mps