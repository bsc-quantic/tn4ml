"""Test Model class methods."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax
import quimb.tensor as qtn
from tn4ml.models.model import Model, _batch_iterator
from tn4ml.models.mps import MPS_initialize
from tn4ml.models.smpo import SMPO_initialize
from tn4ml.embeddings import TrigonometricEmbedding
from tn4ml.initializers import randn
from tn4ml.util import TrainingType, EarlyStopping
import tn4ml.metrics as metrics

jax.config.update("jax_enable_x64", True)


# --- nparams ---


def test_nparams():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=5,
        initializer=randn(1e-2),
        key=key,
        shape_method="even",
        bond_dim=3,
        phys_dim=2,
        cyclic=False,
    )
    n = model.nparams()
    assert n > 0
    assert isinstance(n, (int, np.integer))


# --- configure ---


def test_configure_global():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=5,
        initializer=randn(1e-2),
        key=key,
        shape_method="even",
        bond_dim=3,
        phys_dim=2,
        cyclic=False,
    )
    model.configure(
        strategy="global",
        optimizer=optax.adam,
        learning_rate=0.01,
        loss=metrics.NegLogLikelihood,
        train_type=TrainingType.UNSUPERVISED,
    )
    assert model.strategy == "global"


def test_configure_sweeps():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=5,
        initializer=randn(1e-2),
        key=key,
        shape_method="even",
        bond_dim=3,
        phys_dim=2,
        cyclic=False,
    )
    model.configure(
        strategy="sweeps",
        optimizer=optax.adam,
        learning_rate=0.01,
        loss=metrics.NegLogLikelihood,
        train_type=TrainingType.UNSUPERVISED,
    )
    from tn4ml.strategy import Sweeps

    assert isinstance(model.strategy, Sweeps)


def test_configure_sweeps_one_way():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=5,
        initializer=randn(1e-2),
        key=key,
        shape_method="even",
        bond_dim=3,
        phys_dim=2,
        cyclic=False,
    )
    model.configure(
        strategy="sweeps-one-way",
        optimizer=optax.adam,
        learning_rate=0.01,
        loss=metrics.NegLogLikelihood,
        train_type=TrainingType.UNSUPERVISED,
    )
    from tn4ml.strategy import Sweeps

    assert isinstance(model.strategy, Sweeps)
    assert model.strategy.two_way is False
    assert model.strategy.grouping == 2


def test_configure_sweeps_aliases():
    key = jax.random.PRNGKey(42)
    from tn4ml.strategy import Sweeps

    for alias in ["local", "dmrg", "dmrg-like"]:
        model = MPS_initialize(
            L=5,
            initializer=randn(1e-2),
            key=key,
            shape_method="even",
            bond_dim=3,
            phys_dim=2,
            cyclic=False,
        )
        model.configure(
            strategy=alias,
            optimizer=optax.adam,
            learning_rate=0.01,
            loss=metrics.NegLogLikelihood,
            train_type=TrainingType.UNSUPERVISED,
        )
        assert isinstance(model.strategy, Sweeps), (
            f"alias '{alias}' should produce Sweeps"
        )
        assert model.strategy.two_way is True
        assert model.strategy.grouping == 2


def test_configure_invalid_strategy():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=5,
        initializer=randn(1e-2),
        key=key,
        shape_method="even",
        bond_dim=3,
        phys_dim=2,
        cyclic=False,
    )
    with pytest.raises(ValueError, match="not found"):
        model.configure(
            strategy="invalid_strategy",
            optimizer=optax.adam,
            learning_rate=0.01,
            loss=metrics.NegLogLikelihood,
            train_type=TrainingType.UNSUPERVISED,
        )


def test_configure_invalid_attribute():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=5,
        initializer=randn(1e-2),
        key=key,
        shape_method="even",
        bond_dim=3,
        phys_dim=2,
        cyclic=False,
    )
    with pytest.raises(AttributeError, match="not found"):
        model.configure(nonexistent_param=True)


def test_configure_with_gradient_transforms():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=5,
        initializer=randn(1e-2),
        key=key,
        shape_method="even",
        bond_dim=3,
        phys_dim=2,
        cyclic=False,
    )
    model.configure(
        strategy="global",
        gradient_transforms=[optax.clip_by_global_norm(1.0), optax.adam(1e-3)],
        loss=metrics.NegLogLikelihood,
        train_type=TrainingType.UNSUPERVISED,
    )


def test_configure_invalid_device():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=5,
        initializer=randn(1e-2),
        key=key,
        shape_method="even",
        bond_dim=3,
        phys_dim=2,
        cyclic=False,
    )
    with pytest.raises(AttributeError, match="Device"):
        model.configure(
            strategy="global",
            optimizer=optax.adam,
            learning_rate=0.01,
            loss=metrics.NegLogLikelihood,
            train_type=TrainingType.UNSUPERVISED,
            device=("tpu", 0),
        )


# --- convert_to_pytree ---


def test_convert_to_pytree():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=5,
        initializer=randn(1e-2),
        key=key,
        shape_method="even",
        bond_dim=3,
        phys_dim=2,
        cyclic=False,
    )
    params, skeleton = model.convert_to_pytree()
    assert params is not None
    assert skeleton is not None


# --- _batch_iterator ---


def test_batch_iterator_x_only():
    x = np.random.rand(10, 4)
    batches = list(_batch_iterator(x, batch_size=5, shuffle=False))
    assert len(batches) == 2


def test_batch_iterator_x_and_y():
    x = np.random.rand(10, 4)
    y = np.random.rand(10, 2)
    batches = list(_batch_iterator(x, y, batch_size=5, shuffle=False))
    assert len(batches) == 2
    # Each batch should be a tuple of (x_batch, y_batch)
    assert len(batches[0]) == 2


def test_batch_iterator_shuffle():
    x = np.arange(20).reshape(10, 2)
    batches1 = list(_batch_iterator(x, batch_size=5, shuffle=True, seed=42))
    batches2 = list(_batch_iterator(x, batch_size=5, shuffle=True, seed=0))
    # Different seeds should give different shuffles
    assert not np.array_equal(batches1[0], batches2[0])


# --- predict ---


def test_predict_smpo():
    key = jax.random.PRNGKey(42)
    model = SMPO_initialize(
        L=5,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        spacing=5,
        bond_dim=3,
        phys_dim=(2, 2),
        cyclic=False,
    )
    sample = np.random.rand(5)
    embedding = TrigonometricEmbedding()
    result = model.predict(sample, embedding=embedding)
    assert result is not None


def test_predict_input_too_short():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=10,
        initializer=randn(1e-2),
        key=key,
        shape_method="even",
        bond_dim=3,
        phys_dim=2,
        cyclic=False,
    )
    sample = np.random.rand(5)
    with pytest.raises(ValueError, match="at least"):
        model.predict(sample)


# --- train + evaluate (small integration test) ---


def test_train_unsupervised_global():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=4,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=2,
        phys_dim=2,
        cyclic=False,
    )
    model.configure(
        strategy="global",
        optimizer=optax.adam,
        learning_rate=1e-2,
        loss=metrics.NegLogLikelihood,
        train_type=TrainingType.UNSUPERVISED,
    )
    data = np.random.rand(8, 4)
    history = model.train(
        inputs=data, batch_size=4, epochs=2, embedding=TrigonometricEmbedding()
    )
    assert "loss" in history
    assert len(history["loss"]) == 2


def test_evaluate_unsupervised():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=4,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=2,
        phys_dim=2,
        cyclic=False,
    )
    model.configure(
        strategy="global",
        optimizer=optax.adam,
        learning_rate=1e-2,
        loss=metrics.NegLogLikelihood,
        train_type=TrainingType.UNSUPERVISED,
    )
    data = np.random.rand(4, 4)
    model.batch_size = 4
    loss_val = model.evaluate(
        inputs=data,
        batch_size=4,
        embedding=TrigonometricEmbedding(),
        evaluate_type=TrainingType.UNSUPERVISED,
        metric=metrics.NegLogLikelihood,
    )
    assert isinstance(loss_val, float)


def test_train_sweeps_two_way():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=4,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=2,
        phys_dim=2,
        cyclic=False,
    )
    model.configure(
        strategy="sweeps",
        optimizer=optax.adam,
        learning_rate=1e-2,
        loss=metrics.NegLogLikelihood,
        train_type=TrainingType.UNSUPERVISED,
    )
    data = np.random.rand(8, 4)
    history = model.train(
        inputs=data, batch_size=4, epochs=2, embedding=TrigonometricEmbedding()
    )
    assert "loss" in history
    assert len(history["loss"]) == 2
    assert all(np.isfinite(v) for v in history["loss"])


def test_train_sweeps_one_way():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(
        L=4,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=2,
        phys_dim=2,
        cyclic=False,
    )
    model.configure(
        strategy="sweeps-one-way",
        optimizer=optax.adam,
        learning_rate=1e-2,
        loss=metrics.NegLogLikelihood,
        train_type=TrainingType.UNSUPERVISED,
    )
    data = np.random.rand(8, 4)
    history = model.train(
        inputs=data, batch_size=4, epochs=2, embedding=TrigonometricEmbedding()
    )
    assert "loss" in history
    assert len(history["loss"]) == 2
    assert all(np.isfinite(v) for v in history["loss"])


def test_train_sweeps_opt_states_indexed_by_site():
    """Optimizer state must be keyed per site, not per sweep iteration.
    Verifies the fix where opt_states[opt_index] replaced opt_states[s].
    A two-epoch run would accumulate wrong momentum if states were mixed up."""
    key = jax.random.PRNGKey(0)
    model = MPS_initialize(
        L=4,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=2,
        phys_dim=2,
        cyclic=False,
    )
    model.configure(
        strategy="sweeps",
        optimizer=optax.adam,
        learning_rate=1e-2,
        loss=metrics.NegLogLikelihood,
        train_type=TrainingType.UNSUPERVISED,
    )
    data = np.random.rand(8, 4)
    history = model.train(
        inputs=data, batch_size=8, epochs=3, embedding=TrigonometricEmbedding()
    )
    # Loss should be finite and not NaN throughout (state corruption typically causes NaN)
    assert all(np.isfinite(v) for v in history["loss"])
