import itertools
from numbers import Number
from typing import Callable
import numpy as np
import quimb.tensor as qtn


def trigonometric(x: Number, k: int = 1, dtype=np.float32) -> np.ndarray:
    return 1 / np.sqrt(k) * np.fromiter((f(np.pi * x / 2**i) for f, i in itertools.product([np.cos, np.sin], range(1, k + 1))), dtype=dtype)


def fourier(x: Number, p: int = 2, dtype=np.float32) -> np.ndarray:
    return 1 / p * np.fromiter((np.abs(sum((np.exp(1j * 2 * np.pi * k * ((p - 1) * x - j) / p) for k in range(p)))) for j in range(p)), dtype=dtype)


def embed(x: np.ndarray, /, phi: Callable, **kwargs):
    assert phi.ndim == 1

    arrays = [phi(xi, **kwargs) for xi in x]
    return qtn.MatrixProductState(arrays)
