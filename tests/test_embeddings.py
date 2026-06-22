import itertools

import jax.numpy as jnp
import numpy as np
import pytest

import tn4ml


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 0.3, 0.7, 2.0])
def test_TrigonometricEmbedding(x):  # noqa: N802
    """Test TrigonometricEmbedding."""
    embedding = tn4ml.embeddings.TrigonometricEmbedding()
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), (
        "Norm of embedded data should be 1.0"
    )


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 0.3, 0.7, 2.0])
def test_fourier_embedding_basic(x):
    """Test fourier embedding basic."""
    embedding = tn4ml.embeddings.FourierEmbedding()
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), (
        "Norm of embedded data should be 1.0"
    )


@pytest.mark.parametrize("x", [0.003, 1.45, 2.998, 0.332, 0.3984, 4.83, 6.0])
def test_trigonometric_embedding_extended_range(x):
    """Test trigonometric embedding extended range."""
    embedding = tn4ml.embeddings.TrigonometricEmbedding()
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), (
        "Norm of embedded data should be 1.0"
    )


@pytest.mark.parametrize("x", [0.003, 1.45, 2.998, 0.332, 0.3984, 4.83, 6.0])
def test_fourier_embedding_extended_range(x):
    """Test fourier embedding extended range."""
    embedding = tn4ml.embeddings.FourierEmbedding()
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), (
        "Norm of embedded data should be 1.0"
    )


@pytest.mark.parametrize(
    ("x", "centers", "gamma"),
    [(1, [3], 1), (2, [3], 1), (3, [3], 1), (4, [3], 1), (5, [3], 1)],
)
def test_gaussian_basic(x, centers, gamma):
    """Test gaussian basic."""
    embedding = tn4ml.embeddings.GaussianRBFEmbedding(centers=centers, gamma=gamma)
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), (
        "Norm of embedded data should be 1.0"
    )


@pytest.mark.parametrize(
    ("x", "centers", "gamma"),
    [
        (1, [3, 7], 0.5),
        (2, [3, 7], 0.5),
        (3, [3, 7], 0.5),
        (4, [3, 7], 0.5),
        (5, [3, 7], 0.5),
        (6, [3, 7], 0.5),
        (7, [3, 7], 0.5),
        (8, [3, 7], 0.5),
        (9, [3, 7], 0.5),
        (10, [3, 7], 0.5),
    ],
)
def test_gaussian_multiple_centers(x, centers, gamma):
    """Test gaussian multiple centers."""
    embedding = tn4ml.embeddings.GaussianRBFEmbedding(centers=centers, gamma=gamma)
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), (
        "Norm of embedded data should be 1.0"
    )


@pytest.mark.parametrize(
    ("x", "centers", "gamma"),
    [
        (1, [3, 5], 10),
        (2, [3, 5], 10),
        (3, [3, 5], 10),
        (4, [3, 5], 10),
        (5, [3, 5], 10),
        (6, [3, 5], 10),
    ],
)
def test_gaussian_high_gamma(x, centers, gamma):
    """Test gaussian high gamma."""
    embedding = tn4ml.embeddings.GaussianRBFEmbedding(centers=centers, gamma=gamma)
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), (
        "Norm of embedded data should be 1.0"
    )


@pytest.mark.parametrize(
    ("x", "centers", "gamma"),
    [
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
        (5, [0], 0.1),
    ],
)
def test_gaussian_low_gamma(x, centers, gamma):
    """Test gaussian low gamma."""
    embedding = tn4ml.embeddings.GaussianRBFEmbedding(centers=centers, gamma=gamma)
    phi: jnp.ndarray = embedding(x)
    assert np.linalg.norm(phi) == pytest.approx(1.0), (
        "Norm of embedded data should be 1.0"
    )


@pytest.mark.parametrize(
    ("x", "embedding"),
    itertools.product(
        (np.random.rand(4) for _ in range(5)),  # noqa: NPY002
        [
            tn4ml.embeddings.TrigonometricEmbedding(),
            tn4ml.embeddings.FourierEmbedding(),
        ],
    ),
)
def test_embed_trig_four(x, embedding):
    """Test embed trig four."""
    phi = tn4ml.embeddings.embed(x, phi=embedding)
    assert phi.norm() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("x", "embedding"),
    [
        (
            np.array([1, 2, 3, 4, 5, 6]),
            tn4ml.embeddings.GaussianRBFEmbedding(np.array([3, 5]), 10),
        ),
        (
            np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
            tn4ml.embeddings.GaussianRBFEmbedding(np.array([0]), 0.1),
        ),
        (
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            tn4ml.embeddings.GaussianRBFEmbedding(np.array([3, 7]), 0.5),
        ),
        (
            np.array([1, 2, 3, 4, 5]),
            tn4ml.embeddings.GaussianRBFEmbedding(np.array([3]), 1),
        ),
    ],
)
def test_embed_gauss(x, embedding):
    """Test embed gauss."""
    # zero entry makes problem if x starts with 0
    phi = tn4ml.embeddings.embed(x, phi=embedding)
    assert phi.norm() == pytest.approx(1.0)


# --- LinearComplementEmbedding ---


@pytest.mark.parametrize(
    ("x", "p"),
    [
        (0.3, 2),
        (0.7, 2),
        (0.5, 3),
        (0.1, 3),
    ],
)
def test_LinearComplementEmbedding(x, p):  # noqa: N802
    """Test LinearComplementEmbedding."""
    embedding = tn4ml.embeddings.LinearComplementEmbedding(p=p)
    phi = embedding(x)
    assert phi.shape == (p,)
    assert np.linalg.norm(phi) == pytest.approx(1.0)


def test_LinearComplementEmbedding_invalid_p():  # noqa: N802
    """Test LinearComplementEmbedding invalid p."""
    with pytest.raises(ValueError):  # noqa: PT011
        tn4ml.embeddings.LinearComplementEmbedding(p=4)


# --- QuantumBasisEmbedding ---


@pytest.mark.parametrize("x", [0, 1])
def test_QuantumBasisEmbedding(x):  # noqa: N802
    """Test QuantumBasisEmbedding."""
    basis = {0: [1.0, 0.0], 1: [0.0, 1.0]}
    embedding = tn4ml.embeddings.QuantumBasisEmbedding(basis=basis)
    phi = embedding(x)
    assert phi.shape == (2,)


# --- LegendreEmbedding ---


@pytest.mark.parametrize(
    ("x", "degree"),
    [
        (0.5, 2),
        (-0.5, 3),
        (0.0, 4),
        (1.0, 2),
    ],
)
def test_LegendreEmbedding(x, degree):  # noqa: N802
    """Test LegendreEmbedding."""
    embedding = tn4ml.embeddings.LegendreEmbedding(degree=degree)
    phi = embedding(x)
    assert phi.shape == (degree + 1,)


# --- LaguerreEmbedding ---


@pytest.mark.parametrize(
    ("x", "degree"),
    [
        (0.5, 2),
        (1.0, 3),
        (2.0, 4),
        (0.1, 2),
    ],
)
def test_LaguerreEmbedding(x, degree):  # noqa: N802
    """Test LaguerreEmbedding."""
    embedding = tn4ml.embeddings.LaguerreEmbedding(degree=degree)
    phi = embedding(x)
    assert phi.shape == (degree + 1,)


# --- HermiteEmbedding ---


@pytest.mark.parametrize(
    ("x", "degree"),
    [
        (0.5, 2),
        (-0.5, 3),
        (0.0, 4),
        (1.0, 2),
    ],
)
def test_HermiteEmbedding(x, degree):  # noqa: N802
    """Test HermiteEmbedding."""
    embedding = tn4ml.embeddings.HermiteEmbedding(degree=degree)
    phi = embedding(x)
    assert phi.shape == (degree + 1,)


# --- JaxArraysEmbedding ---


def test_JaxArraysEmbedding_basic():  # noqa: N802
    """Test JaxArraysEmbedding basic."""
    embedding = tn4ml.embeddings.JaxArraysEmbedding(dim=3, input_dim=3)
    x = jnp.array([1.0, 2.0, 3.0])
    phi = embedding(x)
    assert jnp.allclose(phi, x)


def test_JaxArraysEmbedding_with_bias():  # noqa: N802
    """Test JaxArraysEmbedding with bias."""
    embedding = tn4ml.embeddings.JaxArraysEmbedding(dim=4, add_bias=True, input_dim=3)
    x = jnp.array([1.0, 2.0, 3.0])
    phi = embedding(x)
    assert phi.shape == (4,)
    assert phi[0] == 1.0  # bias term


# --- PolynomialEmbedding ---


@pytest.mark.parametrize(
    ("degree", "n", "include_bias"),
    [
        (1, 2, False),
        (2, 2, False),
        (2, 3, True),
        (3, 1, False),
    ],
)
def test_PolynomialEmbedding(degree, n, include_bias):  # noqa: N802
    """Test PolynomialEmbedding."""
    embedding = tn4ml.embeddings.PolynomialEmbedding(
        degree=degree, n=n, include_bias=include_bias
    )
    x = jnp.ones(n) * 0.5
    phi = embedding(x)
    assert phi.shape == (embedding.dim,)


def test_PolynomialEmbedding_invalid_degree():  # noqa: N802
    """Test PolynomialEmbedding invalid degree."""
    with pytest.raises(ValueError):  # noqa: PT011
        tn4ml.embeddings.PolynomialEmbedding(degree=0, n=2)


# --- TrigonometricEmbeddingChain ---


def test_TrigonometricEmbeddingChain():  # noqa: N802
    """Test TrigonometricEmbeddingChain."""
    embedding = tn4ml.embeddings.TrigonometricEmbeddingChain(k=1, input_shape=(2, 2))
    x = [0.5, 0.7]
    phi = embedding(x)
    assert phi.shape == (4,)  # k*2*input_shape[1] per feature


# --- TrigonometricEmbeddingAvg ---


def test_TrigonometricEmbeddingAvg():  # noqa: N802
    """Test TrigonometricEmbeddingAvg."""
    embedding = tn4ml.embeddings.TrigonometricEmbeddingAvg(k=1, input_shape=(2, 2))
    x = [0.5, 0.7]
    phi = embedding(x)
    assert phi.shape == (2,)  # k*2


# --- embed with different embedding types ---


@pytest.mark.parametrize(
    ("x", "embedding"),
    [
        (
            np.array([0.3, 0.5, 0.7, 0.9]),
            tn4ml.embeddings.LinearComplementEmbedding(p=2),
        ),
        (np.array([0.3, 0.5, 0.7, 0.9]), tn4ml.embeddings.LegendreEmbedding(degree=2)),
        (np.array([0.3, 0.5, 0.7, 0.9]), tn4ml.embeddings.LaguerreEmbedding(degree=2)),
        (np.array([0.3, 0.5, 0.7, 0.9]), tn4ml.embeddings.HermiteEmbedding(degree=2)),
    ],
)
def test_embed_various_embeddings(x, embedding):
    """Test embed various embeddings."""
    phi = tn4ml.embeddings.embed(x, phi=embedding)
    assert phi.norm() == pytest.approx(1.0)


def test_embed_invalid_type():
    """Test embed invalid type."""
    with pytest.raises(TypeError):
        tn4ml.embeddings.embed(np.array([0.5]), phi="not_an_embedding")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "embedding_cls",
    [
        tn4ml.embeddings.TrigonometricEmbeddingChain,
        tn4ml.embeddings.TrigonometricEmbeddingAvg,
    ],
)
def test_embed_complex_embedding_2d(embedding_cls):
    """Test embed complex embedding 2d."""
    # 2D input routes embed() through the ComplexEmbedding branch.
    embedding = embedding_cls(k=1, input_shape=(3, 2))
    x = np.array([[0.3, 0.5], [0.4, 0.6], [0.7, 0.9]])
    mps = tn4ml.embeddings.embed(x, phi=embedding)
    assert mps.norm() == pytest.approx(1.0)
