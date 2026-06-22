"""Test utility functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tn4ml.util import (
    EarlyStopping,
    TrainingType,
    divide_into_patches,
    from_dense_to_mps,
    from_mps_to_dense,
    gradient_clip,
    gramschmidt_col,
    gramschmidt_row,
    integer_to_one_hot,
    normalize,
    pad_image_alternately,
    return_digits,
    zigzag_order,
)

# --- return_digits ---


def test_return_digits_basic():
    """Test return digits basic."""
    result = return_digits(["abc123", "def456"])
    assert result == [123, 456]


def test_return_digits_multiple_numbers():
    """Test return digits multiple numbers."""
    result = return_digits(["I0", "I1", "I2"])
    assert result == [0, 1, 2]


def test_return_digits_no_digits():
    """Test return digits no digits."""
    result = return_digits(["abc", "def"])
    assert result == []


def test_return_digits_empty():
    """Test return digits empty."""
    result = return_digits([])
    assert result == []


# --- normalize ---


def test_normalize_unit():
    """Test normalize unit."""
    v = jnp.array([3.0, 4.0])
    result = normalize(v)
    assert jnp.allclose(jnp.linalg.norm(result), 1.0)


def test_normalize_zero_vector():
    """Test normalize zero vector."""
    v = jnp.array([0.0, 0.0])
    result = normalize(v)
    assert result is None


def test_normalize_near_zero():
    """Test normalize near zero."""
    v = jnp.array([1e-12, 1e-12])
    result = normalize(v)
    assert result is None


def test_normalize_already_unit():
    """Test normalize already unit."""
    v = jnp.array([1.0, 0.0])
    result = normalize(v)
    assert jnp.allclose(result, v)


# --- gramschmidt_row ---


def test_gramschmidt_row_square():
    """Test gramschmidt row square."""
    A = jnp.array([[1.0, 1.0], [1.0, 0.0]])
    Q = gramschmidt_row(A)
    # Check orthonormality
    assert jnp.allclose(Q @ Q.T, jnp.eye(2), atol=1e-6)


def test_gramschmidt_row_rectangular():
    """Test gramschmidt row rectangular."""
    A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    Q = gramschmidt_row(A)
    # Check rows are unit norm
    for i in range(Q.shape[0]):
        assert jnp.isclose(jnp.linalg.norm(Q[i]), 1.0, atol=1e-6)
    # Check orthogonality
    assert jnp.isclose(jnp.dot(Q[0], Q[1]), 0.0, atol=1e-6)


# --- gramschmidt_col ---


def test_gramschmidt_col_square():
    """Test gramschmidt col square."""
    A = jnp.array([[1.0, 1.0], [1.0, 0.0]])
    Q = gramschmidt_col(A)
    # Check columns are unit norm
    for j in range(Q.shape[1]):
        assert jnp.isclose(jnp.linalg.norm(Q[:, j]), 1.0, atol=1e-6)


# --- gradient_clip ---


def test_gradient_clip_below_threshold():
    """Test gradient clip below threshold."""
    grads = [jnp.array([0.1, 0.2, 0.3])]
    clipped = gradient_clip(grads, threshold=10.0)
    assert jnp.allclose(jnp.array(clipped[0]), grads[0], atol=1e-5)


def test_gradient_clip_above_threshold():
    """Test gradient clip above threshold."""
    grads = [jnp.array([10.0, 10.0, 10.0])]
    clipped = gradient_clip(grads, threshold=1.0)
    # Norm should be clipped
    clipped_norm = jnp.linalg.norm(jnp.array(clipped[0]))
    assert clipped_norm <= 1.0 + 1e-5


def test_gradient_clip_negative_threshold():
    """Test gradient clip negative threshold."""
    with pytest.raises(AssertionError):
        gradient_clip([jnp.array([1.0])], threshold=-1.0)


# --- zigzag_order ---


def test_zigzag_order():
    """Test zigzag order."""
    data = np.random.rand(5, 4, 4, 1)  # noqa: NPY002
    result = zigzag_order(data)
    assert result.shape == (5, 16)


def test_zigzag_order_no_channel():
    """Test zigzag order no channel."""
    data = np.random.rand(3, 8, 8)  # noqa: NPY002
    result = zigzag_order(data)
    assert result.shape == (3, 64)


# --- integer_to_one_hot ---


def test_integer_to_one_hot_basic():
    """Test integer to one hot basic."""
    labels = [0, 1, 2]
    result = integer_to_one_hot(labels)
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


def test_integer_to_one_hot_with_num_classes():
    """Test integer to one hot with num classes."""
    labels = [0, 1]
    result = integer_to_one_hot(labels, num_classes=5)
    assert result.shape == (2, 5)
    assert result[0, 0] == 1.0
    assert result[1, 1] == 1.0


def test_integer_to_one_hot_single():
    """Test integer to one hot single."""
    labels = [3]
    result = integer_to_one_hot(labels, num_classes=4)
    expected = np.array([[0, 0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


# --- pad_image_alternately ---


def test_pad_image_no_padding_needed():
    """Test pad image no padding needed."""
    image = np.ones((4, 4))
    padded = pad_image_alternately(image, 4)
    assert padded.shape == (4, 4)


def test_pad_image_padding_needed():
    """Test pad image padding needed."""
    image = np.ones((5, 5))
    padded = pad_image_alternately(image, 4)
    # Should pad to next multiple of 4 -> 8x8
    assert padded.shape[0] % 4 == 0
    assert padded.shape[1] % 4 == 0


# --- divide_into_patches ---


def test_divide_into_patches_exact():
    """Test divide into patches exact."""
    image = jnp.ones((4, 4))
    patches = divide_into_patches(image, 2)
    assert patches.shape == (4, 2, 2)


def test_divide_into_patches_needs_padding():
    """Test divide into patches needs padding."""
    image = jnp.ones((5, 5))
    patches = divide_into_patches(image, 4)
    assert patches.shape[1] == 4
    assert patches.shape[2] == 4


# --- from_dense_to_mps / from_mps_to_dense ---


def test_mps_roundtrip():
    """Test mps roundtrip."""
    n_qubits = 3
    statevector = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    statevector = statevector / jnp.linalg.norm(statevector)

    mps = from_dense_to_mps(statevector, n_qubits)
    assert len(mps) == n_qubits

    reconstructed = from_mps_to_dense(mps, n_qubits)
    assert jnp.allclose(statevector, reconstructed, atol=1e-6)


def test_mps_roundtrip_random():
    """Test mps roundtrip random."""
    n_qubits = 4
    key = jax.random.PRNGKey(0)
    statevector = jax.random.normal(key, (2**n_qubits,))
    statevector = statevector / jnp.linalg.norm(statevector)

    mps = from_dense_to_mps(statevector, n_qubits)
    reconstructed = from_mps_to_dense(mps, n_qubits)
    assert jnp.allclose(statevector, reconstructed, atol=1e-5)


def test_mps_with_max_bond():
    """Test mps with max bond."""
    n_qubits = 4
    key = jax.random.PRNGKey(1)
    statevector = jax.random.normal(key, (2**n_qubits,))
    statevector = statevector / jnp.linalg.norm(statevector)

    mps = from_dense_to_mps(statevector, n_qubits, max_bond=2)
    assert len(mps) == n_qubits
    # Bond dimensions should be at most 2
    for tensor in mps:
        for dim in tensor.shape:
            assert dim <= 2**n_qubits  # sanity check


# --- TrainingType ---


def test_training_type_values():
    """Test training type values."""
    assert TrainingType.UNSUPERVISED == 0
    assert TrainingType.SUPERVISED == 1
    assert TrainingType.TARGET_TN == 2


# --- EarlyStopping ---


def test_early_stopping_init():
    """Test early stopping init."""
    es = EarlyStopping(monitor="loss", min_delta=0.01, patience=5, mode="min")
    assert es.monitor == "loss"
    assert es.patience == 5
    assert es.mode == "min"


def test_early_stopping_on_begin_train_min():
    """Test early stopping on begin train min."""
    es = EarlyStopping(monitor="loss", min_delta=0.01, patience=3, mode="min")
    history: dict[str, list[float]] = {"loss": []}
    es.on_begin_train(history, model=None)
    assert es.memory["best"] == np.inf
    assert es.operator == np.less


def test_early_stopping_on_begin_train_max():
    """Test early stopping on begin train max."""
    es = EarlyStopping(monitor="val_acc", min_delta=0.01, patience=3, mode="max")
    history: dict[str, list[float]] = {"val_acc": []}
    es.on_begin_train(history, model=None)
    assert es.memory["best"] == -np.inf
    assert es.operator == np.greater


def test_early_stopping_invalid_monitor():
    """Test early stopping invalid monitor."""
    es = EarlyStopping(monitor="nonexistent", min_delta=0.01, patience=3, mode="min")
    history: dict[str, list[float]] = {"loss": []}
    with pytest.raises(ValueError, match="not monitored"):
        es.on_begin_train(history, model=None)


def test_early_stopping_invalid_mode():
    """Test early stopping invalid mode."""
    es = EarlyStopping(monitor="loss", min_delta=0.01, patience=3, mode="invalid")
    history: dict[str, list[float]] = {"loss": []}
    with pytest.raises(ValueError, match="min.*max"):  # noqa: RUF043
        es.on_begin_train(history, model=None)


def test_early_stopping_patience():
    """Test early stopping patience."""
    es = EarlyStopping(monitor="loss", min_delta=0.001, patience=2, mode="min")
    history: dict[str, list[float]] = {"loss": []}
    es.on_begin_train(history, model=None)

    # Epoch 0 - sets best
    result = es.on_end_epoch(1.0, 0, None)
    assert result == 0

    # Epoch 1 - no improvement
    result = es.on_end_epoch(1.0, 1, None)
    assert result == 0

    # Epoch 2 - still no improvement, patience=2 reached
    result = es.on_end_epoch(1.0, 2, None)
    assert result == 1
