"""Test initializer functions"""

import jax
import jax.numpy as jnp
import pytest

import tn4ml
from tn4ml.util import *


def check_orthonormal_vectors(Q, type="rows", atol=1e-6):  # noqa: A002
    """
    Check if the rows of matrix Q form an orthonormal set.

    Parameters:
    Q : numpy.ndarray
        The matrix with vectors as its rows.
    atol : float, optional
        The absolute tolerance for numerical comparisons.

    Returns:
    bool
        True if the rows of Q are orthonormal, False otherwise.
    """
    num_rows, num_cols = Q.shape

    if type == "rows":
        # Check normality (unit norm for each row vector)
        for row in range(num_rows):
            if not jnp.isclose(jnp.linalg.norm(Q[row, :]), 1.0, atol=atol):
                return False
        print("Normality okay")
        # Check orthogonality between each pair of distinct row vectors
        for i in range(num_rows):
            for j in range(i + 1, num_rows):
                print(jnp.dot(Q[i, :], Q[j, :]))
                if not jnp.isclose(jnp.dot(Q[i, :], Q[j, :]), 0, atol=atol):
                    return False

        return True
    # Check normality (unit norm for each col vector)
    for col in range(num_cols):
        print(jnp.linalg.norm(Q[:, col]))
        if not jnp.isclose(jnp.linalg.norm(Q[:, col]), 1.0, atol=atol):
            return False
    print("Normality okay")
    # Check orthogonality between each pair of distinct col vectors
    for i in range(num_cols):
        for j in range(i + 1, num_cols):
            print(jnp.dot(Q[:, i], Q[:, j]))
            if not jnp.isclose(jnp.dot(Q[:, i], Q[:, j]), 0, atol=atol):
                return False

    return True


@pytest.mark.parametrize(
    ("std", "mean", "shape"),
    [
        (1.0, 0.0, (1, 2, 2)),
        (1e-9, 1e-6, (5, 5, 2, 3)),
        (0.5, 1e-6, (5, 5, 2, 3)),
        (1e-9, None, (5, 5, 2)),
        (0.5, None, (5, 5, 2)),
    ],
)
def test_randn_init_with_parameters(std, mean, shape):
    """Test randn init with parameters."""
    initializer = tn4ml.initializers.randn(std, mean)
    Q = initializer(jax.random.key(42), shape, jnp.float32)
    assert Q is not None


@pytest.mark.parametrize(
    ("dist", "scale", "shape"),
    [
        ("uniform", 1e-2, (2, 3)),
        ("uniform", 1e-2, (5, 5)),
        ("uniform", 1e-2, (1, 5)),
        ("normal", 1e-2, (2, 3)),
        ("normal", 1e-2, (5, 5)),
        ("normal", 1e-2, (1, 5)),
        ("uniform", 1e-3, (5, 5, 2)),
        ("normal", 1e-3, (5, 5, 2)),
        ("uniform", 1e-3, (5, 5, 2, 3)),
        ("normal", 1e-3, (5, 5, 2, 3)),
    ],
)
def test_gramschmidt_init(dist, scale, shape):
    """Test gramschmidt init."""
    initializer = tn4ml.initializers.gramschmidt(dist, scale)
    Q = initializer(jax.random.key(42), shape, jnp.float32)
    matrix_shape = shape[0], np.prod(shape[1:])

    Q = Q.reshape(matrix_shape)
    assert check_orthonormal_vectors(Q, "rows", atol=1e-5)
    assert Q is not None


@pytest.mark.parametrize("shape", [((1, 2, 2)), ((5, 5, 2, 3)), ((5, 5, 2))])
def test_randn_init_default(shape):
    """Test randn init default."""
    initializer = tn4ml.initializers.randn()
    Q = initializer(jax.random.key(42), shape, jnp.float32)
    assert Q is not None


# --- zeros ---


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (3, 4, 2),
        (5, 5),
    ],
)
def test_zeros_init(shape):
    """Test zeros init."""
    initializer = tn4ml.initializers.zeros()
    Q = initializer(jax.random.key(42), shape, jnp.float32)
    assert Q.shape == shape
    # Should be close to zero (with small noise)
    assert jnp.allclose(Q, jnp.zeros(shape), atol=1e-7)


# --- ones ---


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (3, 4, 2),
        (5, 5),
    ],
)
def test_ones_init(shape):
    """Test ones init."""
    initializer = tn4ml.initializers.ones()
    Q = initializer(jax.random.key(42), shape, jnp.float32)
    assert Q.shape == shape
    # Should be close to ones (with small noise)
    assert jnp.allclose(Q, jnp.ones(shape), atol=1e-7)


# --- identity (copy mode) ---


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (3, 3),
        (4, 3),
    ],
)
def test_identity_copy(shape):
    """Test identity copy."""
    initializer = tn4ml.initializers.identity("copy", std=1e-3)
    Q = initializer(jax.random.key(42), shape, jnp.float32)
    assert Q.shape == shape
    # Diagonal elements should be close to 1
    for i in range(min(shape)):
        assert jnp.isclose(Q[(i,) * len(shape)], 1.0, atol=0.1)


# --- identity (bond mode) ---


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3, 2),
        (2, 4, 2, 3),
    ],
)
def test_identity_bond(shape):
    """Test identity bond."""
    initializer = tn4ml.initializers.identity("bond")
    Q = initializer(jax.random.key(42), shape, jnp.float32)
    assert Q.shape == shape


def test_identity_invalid_type():
    """Test identity invalid type."""
    with pytest.raises(ValueError, match="Defined only"):  # noqa: PT012
        initializer = tn4ml.initializers.identity("invalid")
        initializer(jax.random.key(42), (3, 3), jnp.float32)


# --- unitary_matrix ---


def test_unitary_matrix():
    """Test unitary matrix."""
    Q = tn4ml.initializers.unitary_matrix(jax.random.key(42), (4, 4), jnp.float32)
    # Q @ Q^T should be identity
    assert jnp.allclose(Q @ Q.T, jnp.eye(4), atol=1e-5)


def test_unitary_matrix_non_square():
    """Test unitary matrix non square."""
    with pytest.raises(AssertionError):
        tn4ml.initializers.unitary_matrix(jax.random.key(42), (3, 4), jnp.float32)


# --- rand_unitary ---


def test_rand_unitary_init():
    """Test rand unitary init."""
    # rand_unitary internally uses shape[2] so requires a 3D shape
    initializer = tn4ml.initializers.rand_unitary()
    Q = initializer(jax.random.key(42), (3, 3, 2), jnp.float32)
    assert Q.shape == (3, 3, 2)
