import numpy as np
import pytest
import tnad
import itertools


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 0.3, 0.7, 2.0])
def test_trigonometric(x):
    embedding = tnad.embeddings.trigonometric()
    phi: np.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0)


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 0.3, 0.7, 2.0])
def test_fourier(x):
    embedding = tnad.embeddings.fourier()
    phi: np.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0)


@pytest.mark.parametrize("x,embedding", itertools.product((np.random.rand(4) for _ in range(5)), [tnad.embeddings.trigonometric(), tnad.embeddings.fourier()]))
def test_embed(x, embedding):
    phi = tnad.embeddings.embed(x, phi=embedding)
    assert phi.norm() == pytest.approx(1.0)
