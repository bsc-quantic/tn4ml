"""Test Model class methods."""

import jax
import numpy as np
import optax
import pytest

import tn4ml.metrics as metrics
from tn4ml.embeddings import TrigonometricEmbedding
from tn4ml.initializers import randn
from tn4ml.models.model import Model, _batch_iterator, load_model
from tn4ml.models.mps import MPS_initialize
from tn4ml.models.smpo import SMPO_initialize
from tn4ml.models.tn import TN_initialize
from tn4ml.util import TrainingType

jax.config.update("jax_enable_x64", True)


class _IdentityAccuracyModel(Model):
    def __init__(self):
        self.device = ("cpu", 0)

    def predict(self, sample, embedding=None, return_tn=False, normalize=False):
        return sample


# --- nparams ---


def test_nparams():
    """Test nparams."""
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


# --- save / load ---


def test_save_and_load_model(tmp_path):
    """Test save and load model."""
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
    model.save("mymodel", dir_name=str(tmp_path))
    assert (tmp_path / "mymodel.pkl").exists()

    loaded = load_model("mymodel", dir_name=str(tmp_path))
    assert loaded.nparams() == model.nparams()
    assert loaded.norm() == pytest.approx(model.norm(), abs=1e-6)


def test_save_with_tn_flag(tmp_path):
    """Test save with tn flag."""
    # tn=True is the documented path for generic TensorNetwork models, which are
    # rebuilt from their tensors/inds on save.
    key = jax.random.PRNGKey(0)
    shapes = [(3, 2), (3, 3, 2), (3, 2)]
    inds = [["bond_0", "k0"], ["bond_0", "bond_1", "k1"], ["bond_1", "k2"]]
    model = TN_initialize(shapes=shapes, key=key, initializer=randn(1e-2), inds=inds)
    model.save("tnmodel", dir_name=str(tmp_path), tn=True)
    assert (tmp_path / "tnmodel.pkl").exists()

    loaded = load_model("tnmodel", dir_name=str(tmp_path))
    assert loaded.nparams() == model.nparams()


# --- update_tensors ---


def test_update_tensors_global():
    """Test update tensors global."""
    key = jax.random.PRNGKey(1)
    model = MPS_initialize(
        L=4,
        initializer=randn(1e-2),
        key=key,
        shape_method="even",
        bond_dim=3,
        phys_dim=2,
        cyclic=False,
    )
    model.strategy = "global"
    new_params = [t.data * 2.0 for t in model.tensors]
    model.update_tensors(new_params)
    np.testing.assert_allclose(model.tensors[0].data, new_params[0])


# --- compute_entropy ---


def test_compute_entropy():
    """Test compute entropy."""
    key = jax.random.PRNGKey(2)
    model = SMPO_initialize(
        L=8,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        cyclic=False,
    )
    data = np.random.rand(8)  # noqa: NPY002
    entropy = model.compute_entropy(data, TrigonometricEmbedding())
    assert np.isfinite(float(entropy))


def test_compute_entropy_batch():
    """Test compute entropy batch."""
    key = jax.random.PRNGKey(3)
    model = SMPO_initialize(
        L=8,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        cyclic=False,
    )
    data = np.random.rand(3, 8)  # noqa: NPY002
    entropy = model.compute_entropy_batch(data, TrigonometricEmbedding())
    assert np.isfinite(float(entropy))


# --- configure ---


def test_configure_global():
    """Test configure global."""
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
    """Test configure sweeps."""
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
    """Test configure sweeps one way."""
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
    """Test configure sweeps aliases."""
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
    """Test configure invalid strategy."""
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
    """Test configure invalid attribute."""
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
    """Test configure with gradient transforms."""
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
    """Test configure invalid device."""
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
    """Test convert to pytree."""
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
    """Test batch iterator x only."""
    x = np.random.rand(10, 4)  # noqa: NPY002
    batches = list(_batch_iterator(x, batch_size=5, shuffle=False))
    assert len(batches) == 2


def test_batch_iterator_x_and_y():
    """Test batch iterator x and y."""
    x = np.random.rand(10, 4)  # noqa: NPY002
    y = np.random.rand(10, 2)  # noqa: NPY002
    batches = list(_batch_iterator(x, y, batch_size=5, shuffle=False))
    assert len(batches) == 2
    # Each batch should be a tuple of (x_batch, y_batch)
    assert len(batches[0]) == 2


def test_batch_iterator_shuffle():
    """Test batch iterator shuffle."""
    x = np.arange(20).reshape(10, 2)
    batches1 = list(_batch_iterator(x, batch_size=5, shuffle=True, seed=42))
    batches2 = list(_batch_iterator(x, batch_size=5, shuffle=True, seed=0))
    # Different seeds should give different shuffles
    assert not np.array_equal(batches1[0], batches2[0])


# --- predict ---


def test_predict_smpo():
    """Test predict smpo."""
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
    sample = np.random.rand(5)  # noqa: NPY002
    embedding = TrigonometricEmbedding()
    result = model.predict(sample, embedding=embedding)
    assert result is not None


def test_predict_input_too_short():
    """Test predict input too short."""
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
    sample = np.random.rand(5)  # noqa: NPY002
    with pytest.raises(ValueError, match="at least"):
        model.predict(sample)


# --- accuracy ---


def test_accuracy_uses_raw_outputs_by_default():
    """Test accuracy uses raw outputs by default."""
    model = _IdentityAccuracyModel()
    data = np.array([[0.1, 0.9], [0.8, 0.2]])
    targets = np.array([[0, 1], [1, 0]])

    accuracy = model.accuracy(data, targets, batch_size=2)

    assert accuracy == pytest.approx(1.0)


def test_accuracy_accepts_score_transform():
    """Test accuracy accepts score transform."""
    model = _IdentityAccuracyModel()
    data = np.array([[0.1, 0.9], [0.8, 0.2]])
    targets = np.array([[0, 1], [1, 0]])

    accuracy = model.accuracy(
        data,
        targets,
        batch_size=2,
        accuracy_fn=lambda scores: scores[:, ::-1],
    )

    assert accuracy == pytest.approx(0.0)


def test_accuracy_accepts_label_transform_and_integer_targets():
    """Test accuracy accepts label transform and integer targets."""
    model = _IdentityAccuracyModel()
    data = np.array([[0.1], [0.9]])
    targets = np.array([0, 1])

    accuracy = model.accuracy(
        data,
        targets,
        batch_size=2,
        accuracy_fn=lambda scores: scores[:, 0] > 0.5,
    )

    assert accuracy == pytest.approx(1.0)


# --- train + evaluate (small integration test) ---


def test_train_unsupervised_global():
    """Test train unsupervised global."""
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
    data = np.random.rand(8, 4)  # noqa: NPY002
    history = model.train(
        inputs=data, batch_size=4, epochs=2, embedding=TrigonometricEmbedding()
    )
    assert "loss" in history
    assert len(history["loss"]) == 2


def test_evaluate_unsupervised():
    """Test evaluate unsupervised."""
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
    data = np.random.rand(4, 4)  # noqa: NPY002
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
    """Test train sweeps two way."""
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
    data = np.random.rand(8, 4)  # noqa: NPY002
    history = model.train(
        inputs=data, batch_size=4, epochs=2, embedding=TrigonometricEmbedding()
    )
    assert "loss" in history
    assert len(history["loss"]) == 2
    assert all(np.isfinite(v) for v in history["loss"])


def test_train_sweeps_one_way():
    """Test train sweeps one way."""
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
    data = np.random.rand(8, 4)  # noqa: NPY002
    history = model.train(
        inputs=data, batch_size=4, epochs=2, embedding=TrigonometricEmbedding()
    )
    assert "loss" in history
    assert len(history["loss"]) == 2
    assert all(np.isfinite(v) for v in history["loss"])


def test_train_sweeps_opt_states_indexed_by_site():
    """Optimizer state must be keyed by ordered sweep step.

    Forward and backward passes over the same bond can produce different
    contracted tensor axis orders, so they cannot share one Adam state.
    """
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
    data = np.random.rand(8, 4)  # noqa: NPY002
    history = model.train(
        inputs=data, batch_size=8, epochs=3, embedding=TrigonometricEmbedding()
    )
    # Loss should be finite and not NaN throughout (state corruption typically causes NaN)
    assert all(np.isfinite(v) for v in history["loss"])
    assert set(model.opt_states) == set(model.strategy.iterate_sites(model))
    assert (1, 2) in model.opt_states
    assert (2, 1) in model.opt_states
