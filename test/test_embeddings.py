import numpy as np
import pytest
import tnad


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 0.3, 0.7, 2.0])
def test_trigonometric(x):
    phi: np.ndarray = tnad.embeddings.trigonometric(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0)


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 0.3, 0.7, 2.0])
def test_fourier(x):
    phi: np.ndarray = tnad.embeddings.fourier(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0)
