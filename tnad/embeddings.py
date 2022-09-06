import itertools
from numbers import Number
import numpy as np
import quimb.tensor as qtn
import abc


class Embedding:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        pass

    @abc.abstractmethod
    def __call__(self, x: Number) -> np.ndarray:
        pass


class trigonometric(Embedding):
    def __init__(self, k: int = 1, **kwargs):
        assert k >= 1

        self.k = 1
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        return self.k * 2

    def __call__(self, x: Number) -> np.ndarray:
        return 1 / np.sqrt(self.k) * np.fromiter((f(np.pi * x / 2**i) for f, i in itertools.product([np.cos, np.sin], range(1, self.k + 1))), dtype=self.dtype)


class fourier(Embedding):
    def __init__(self, p: int = 2, **kwargs):
        assert p >= 2

        self.p = 2
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        return self.p

    def __call__(self, x: Number) -> np.ndarray:
        return 1 / self.p * np.fromiter((np.abs(sum((np.exp(1j * 2 * np.pi * k * ((self.p - 1) * x - j) / self.p) for k in range(self.p)))) for j in range(self.p)), dtype=self.dtype)


def embed(x: np.ndarray, phi: Embedding, **mps_opts):
    """Creates a product state from a vector of features `x`."""
    assert x.ndim == 1

    arrays = [phi(xi).reshape((1, 1, phi.dim)) for xi in x]

    return qtn.MatrixProductState(arrays, **mps_opts)
