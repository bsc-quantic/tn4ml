import numpy as np
import jax.numpy as jnp
import pytest
import tn4ml
import itertools

@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 0.3, 0.7, 2.0])
def test_trigonometric(x):
    embedding = tn4ml.embeddings.trigonometric()
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), "Norm of embedded data should be 1.0"

@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 0.3, 0.7, 2.0])
def test_fourier(x):
    embedding = tn4ml.embeddings.fourier()
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), "Norm of embedded data should be 1.0"

@pytest.mark.parametrize("x", [0.003, 1.45, 2.998, 0.332, 0.3984, 4.83, 6.0])
def test_fourier(x):
    embedding = tn4ml.embeddings.trigonometric()
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), "Norm of embedded data should be 1.0"

@pytest.mark.parametrize("x", [0.003, 1.45, 2.998, 0.332, 0.3984, 4.83, 6.0])
def test_fourier(x):
    embedding = tn4ml.embeddings.fourier()
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), "Norm of embedded data should be 1.0"

@pytest.mark.parametrize("x", [0.003, 1.45, 2.998, 0.332, 0.3984, 4.83, 6.0])
def test_linear(x):
    embedding = tn4ml.embeddings.linear()
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), "Norm of embedded data should be 1.0"

@pytest.mark.parametrize("x,centers,gamma", [
    (1, [3], 1),
    (2, [3], 1),
    (3, [3], 1),
    (4, [3], 1),
    (5, [3], 1)
])
def test_gaussian_basic(x, centers, gamma):
    embedding = tn4ml.embeddings.gaussian_rbf(centers=centers, gamma=gamma)
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), "Norm of embedded data should be 1.0"

@pytest.mark.parametrize("x,centers,gamma", [
    (1, [3,7], 0.5),
    (2, [3,7], 0.5),
    (3, [3,7], 0.5),
    (4, [3,7], 0.5),
    (5, [3,7], 0.5),
    (6, [3,7], 0.5),
    (7, [3,7], 0.5),
    (8, [3,7], 0.5),
    (9, [3,7], 0.5),
    (10, [3,7], 0.5)
])
def test_gaussian_multiple_centers(x, centers, gamma):
    embedding = tn4ml.embeddings.gaussian_rbf(centers=centers, gamma=gamma)
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), "Norm of embedded data should be 1.0"

@pytest.mark.parametrize("x,centers,gamma", [
    (1, [3,5], 10),
    (2, [3,5], 10),
    (3, [3,5], 10),
    (4, [3,5], 10),
    (5, [3,5], 10),
    (6, [3,5], 10)
])
def test_gaussian_high_gamma(x, centers, gamma):
    embedding = tn4ml.embeddings.gaussian_rbf(centers=centers, gamma=gamma)
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), "Norm of embedded data should be 1.0"

@pytest.mark.parametrize("x,centers,gamma", [
    (-5, [0], 0.1),
    (-4, [0], 0.1),
    (-3, [0], 0.1),
    (-2, [0], 0.1),
    (-1, [0], 0.1),
    (0, [0], 0.1),
    (1, [0], 0.1),
    (2, [0], 0.1),
    (3, [0], 0.1),
    (4, [0], 0.1),
    (5, [0], 0.1)
])
def test_gaussian_low_gamma(x, centers, gamma):
    embedding = tn4ml.embeddings.gaussian_rbf(centers=centers, gamma=gamma)
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), "Norm of embedded data should be 1.0"

@pytest.mark.parametrize("x,embedding", itertools.product((np.random.rand(4) for _ in range(5)), [tn4ml.embeddings.trigonometric(), tn4ml.embeddings.fourier()]))
def test_embed_trig_four(x, embedding):
    phi = tn4ml.embeddings.embed(x, phi=embedding)
    assert phi.norm() == pytest.approx(1.0)

@pytest.mark.parametrize("x,embedding", [
    (np.array([1,2,3,4,5,6]), tn4ml.embeddings.gaussian_rbf([3,5], 10)),
    (np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5]), tn4ml.embeddings.gaussian_rbf([0], 0.1)),
    (np.array([1,2,3,4,5,6,7,8,9,10]), tn4ml.embeddings.gaussian_rbf([3,7], 0.5)),
    (np.array([1,2,3,4,5]), tn4ml.embeddings.gaussian_rbf([3], 1))
])
def test_embed_gauss(x, embedding):
    # zero entry makes problem if x starts with 0
    phi = tn4ml.embeddings.embed(x, phi=embedding)
    assert phi.norm() == pytest.approx(1.0)

@pytest.mark.parametrize("x,embedding", [
    (np.array([[0,1,2], [3,4,5]]), tn4ml.embeddings.trigonometric_chain(input_shape=(2,3))),
    (np.array([[0,1,2], [3,4,5]]), tn4ml.embeddings.trigonometric_avg(input_shape=(2,3))),
])

def test_embed_trig_chain_avg(x, embedding):
    phi = tn4ml.embeddings.embed(x, phi=embedding)
    assert phi.norm() == pytest.approx(1.0)