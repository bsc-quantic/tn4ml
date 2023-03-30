import numpy as np
import pytest
import tn4ml
import itertools


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 0.3, 0.7, 2.0])
def test_trigonometric(x):
    embedding = tn4ml.embeddings.trigonometric()
    phi: np.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0)


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 0.3, 0.7, 2.0])
def test_fourier(x):
    embedding = tn4ml.embeddings.fourier()
    phi: np.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0)


@pytest.mark.parametrize("x,embedding", itertools.product((np.random.rand(4) for _ in range(5)), [tn4ml.embeddings.trigonometric(), tn4ml.embeddings.fourier()]))
def test_embed(x, embedding):
    phi = tn4ml.embeddings.embed(x, phi=embedding)
    assert phi.norm() == pytest.approx(1.0)
