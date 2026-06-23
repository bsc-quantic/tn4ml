import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import quimb.tensor as qtn

import tn4ml
import tn4ml.embeddings as embeddings
import tn4ml.metrics as metrics
import tn4ml.models.smpo as smpo
from tn4ml.initializers import randn


def _smpo(L=10, spacing=2, phys_dim=(2, 2)):
    return smpo.SMPO_initialize(
        L=L,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        spacing=spacing,
        bond_dim=4,
        phys_dim=phys_dim,
        cyclic=False,
    )


def _mps(L=10, phys_dim=2):
    return tn4ml.models.mps.MPS_initialize(
        L=L,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        bond_dim=4,
        phys_dim=phys_dim,
        cyclic=False,
    )


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (
            qtn.MPS_rand_state(20, bond_dim=2, phys_dim=2),
            qtn.MPS_rand_state(20, bond_dim=5, phys_dim=2),
        )
    ],
)
def test_NegLogLikelihood_1(model, data):  # noqa: N802
    """Test NegLogLikelihood 1."""
    loss = tn4ml.metrics.NegLogLikelihood(model, data)
    assert loss >= 0.0
    # check if physical dimensions match
    assert model.tensors[0].shape[-1] == data.tensors[0].shape[-1]
    assert len(model.tensors) == len(data.tensors)


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (
            qtn.MPS_rand_state(20, bond_dim=2, phys_dim=2, cyclic=True),
            qtn.MPS_rand_state(20, bond_dim=5, phys_dim=2),
        )
    ],
)
def test_NegLogLikelihood_2(model, data):  # noqa: N802
    """Test NegLogLikelihood 2."""
    loss = tn4ml.metrics.NegLogLikelihood(model, data)
    assert loss >= 0.0
    # check if physical dimensions match
    assert model.tensors[0].shape[-1] == data.tensors[0].shape[-1]
    assert len(model.tensors) == len(data.tensors)


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (
            qtn.MPS_rand_state(20, bond_dim=2, phys_dim=2, cyclic=True),
            qtn.MPS_rand_state(20, bond_dim=5, phys_dim=2, cyclic=True),
        )
    ],
)
def test_NegLogLikelihood_3(model, data):  # noqa: N802
    """Test NegLogLikelihood 3."""
    loss = tn4ml.metrics.NegLogLikelihood(model, data)
    assert loss >= 0.0
    # check if physical dimensions match
    assert model.tensors[0].shape[-1] == data.tensors[0].shape[-1]
    assert len(model.tensors) == len(data.tensors)


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (
            qtn.MPS_rand_state(10, bond_dim=2, phys_dim=3),
            np.random.rand(  # noqa: NPY002
                10,
            ),
        )
    ],
)
def test_NegLogLikelihood_4(model, data):  # noqa: N802
    """Test NegLogLikelihood 4."""
    embedding = tn4ml.embeddings.FourierEmbedding(p=3)
    phi = tn4ml.embeddings.embed(data, phi=embedding)
    loss = tn4ml.metrics.NegLogLikelihood(model, phi)
    assert loss >= 0.0
    # check if physical dimensions match
    assert model.tensors[0].shape[-1] == phi.tensors[0].shape[-1]
    assert len(model.tensors) == len(phi.tensors)


# TODO - when SMPO is fully working!
@pytest.mark.parametrize(
    ("model", "data"),
    [
        (
            tn4ml.models.smpo.SMPO_initialize(
                L=10,
                initializer=randn(1e-1),
                key=jax.random.key(42),
                shape_method="even",
                spacing=2,
                bond_dim=4,
                phys_dim=(2, 2),
                cyclic=False,
            ),
            np.random.rand(  # noqa: NPY002
                10,
            ),
        )
    ],
)
def test_transformed_squared_norm(model, data):
    """Test transformed squared norm."""
    embedding = tn4ml.embeddings.TrigonometricEmbedding()
    phi = tn4ml.embeddings.embed(data, phi=embedding)
    loss = tn4ml.metrics.TransformedSquaredNorm(model, phi)
    assert loss >= 0.0
    # check if physical dimensions match
    assert model.tensors[0].shape[-1] == phi.tensors[0].shape[-1]
    assert len(model.tensors) == len(phi.tensors)


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (
            tn4ml.models.smpo.SMPO_initialize(
                L=10,
                initializer=randn(1e-1),
                key=jax.random.key(42),
                shape_method="even",
                spacing=2,
                bond_dim=4,
                phys_dim=(2, 2),
                cyclic=False,
            ),
            qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2),
        )
    ],
)
def test_transformed_squared_norm_2(model, data):
    """Test transformed squared norm 2."""
    loss = tn4ml.metrics.TransformedSquaredNorm(model, data)
    assert loss >= 0.0
    # check if physical dimensions match
    assert model.tensors[0].shape[-1] == data.tensors[0].shape[-1]
    assert len(model.tensors) == len(data.tensors)


# TODO add more MPO initializations from quimb
@pytest.mark.parametrize(
    "model",
    [
        (qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)),
        (qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2, cyclic=True)),
    ],
)
def test_LogFrobNorm_MPS(model):  # noqa: N802
    """Test LogFrobNorm MPS."""
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)


# TODO add more MPO initializations from quimb
@pytest.mark.parametrize(
    "model",
    [
        (qtn.MPO_rand(10, bond_dim=2, phys_dim=2)),
        (qtn.MPO_rand(10, bond_dim=2, phys_dim=2, cyclic=True)),
    ],
)
def test_LogFrobNorm_MPO(model):  # noqa: N802
    """Test LogFrobNorm MPO."""
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)


@pytest.mark.parametrize(
    "model",
    [
        (
            tn4ml.models.smpo.SMPO_initialize(
                L=10,
                initializer=randn(1e-1),
                key=jax.random.key(42),
                shape_method="even",
                spacing=2,
                bond_dim=4,
                phys_dim=(2, 2),
                cyclic=False,
            )
        ),
        (
            tn4ml.models.smpo.SMPO_initialize(
                L=15,
                initializer=tn4ml.initializers.randn(1e-9, dtype=jnp.float64),
                key=jax.random.key(42),
                shape_method="even",
                spacing=5,
                bond_dim=4,
                phys_dim=(2, 2),
                cyclic=False,
            )
        ),
    ],
)
def test_LogFrobNorm_SMPO(model):  # noqa: N802
    """Test LogFrobNorm SMPO."""
    loss = tn4ml.metrics.LogFrobNorm(model)
    print(jax.device_get(loss))
    assert isinstance(jax.device_get(loss), np.ndarray)


@pytest.mark.parametrize(
    "model",
    [
        (
            tn4ml.models.mps.MPS_initialize(
                L=10,
                initializer=randn(1e-1),
                key=jax.random.key(42),
                shape_method="even",
                bond_dim=4,
                phys_dim=2,
                cyclic=False,
            )
        ),
        (
            tn4ml.models.mps.MPS_initialize(
                L=20,
                initializer=tn4ml.initializers.randn(1e-9, dtype=jnp.float64),
                key=jax.random.key(42),
                shape_method="noteven",
                bond_dim=4,
                phys_dim=2,
                cyclic=False,
            )
        ),
    ],
)
def test_LogFrobNorm_TrainableMPS(model):  # noqa: N802
    """Test LogFrobNorm TrainableMPS."""
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)


@pytest.mark.parametrize(
    "model",
    [
        (
            tn4ml.models.mpo.MPO_initialize(
                L=10,
                initializer=randn(1e-1),
                key=jax.random.key(42),
                shape_method="even",
                bond_dim=4,
                phys_dim=(2, 2),
                cyclic=False,
            )
        ),
        (
            tn4ml.models.mpo.MPO_initialize(
                L=20,
                initializer=tn4ml.initializers.randn(1e-9, dtype=jnp.float64),
                key=jax.random.key(42),
                shape_method="noteven",
                bond_dim=4,
                phys_dim=(2, 2),
                cyclic=False,
            )
        ),
    ],
)
def test_LogFrobNorm_TrainableMP0(model):  # noqa: N802
    """Test LogFrobNorm TrainableMP0."""
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)


# TODO add more MPS initializations from quimb
@pytest.mark.parametrize(
    "model",
    [
        (qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)),
        (qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2, cyclic=True)),
    ],
)
def test_LogReLUFrobNorm_MPS(model):  # noqa: N802
    """Test LogReLUFrobNorm MPS."""
    loss = tn4ml.metrics.LogReLUFrobNorm(model)
    assert jax.device_get(loss) >= 0.0


# TODO add more MPO initializations from quimb
@pytest.mark.parametrize(
    "model",
    [
        (qtn.MPO_rand(10, bond_dim=2, phys_dim=2)),
        (qtn.MPO_rand(10, bond_dim=2, phys_dim=2, cyclic=True)),
    ],
)
def test_LogReLUFrobNorm_MPO(model):  # noqa: N802
    """Test LogReLUFrobNorm MPO."""
    loss = tn4ml.metrics.LogReLUFrobNorm(model)
    assert jax.device_get(loss) >= 0.0


@pytest.mark.parametrize(
    "model",
    [
        (
            tn4ml.models.smpo.SMPO_initialize(
                L=10,
                initializer=randn(1e-1),
                key=jax.random.key(42),
                shape_method="even",
                spacing=2,
                bond_dim=4,
                phys_dim=(2, 2),
                cyclic=False,
            )
        ),
        (
            tn4ml.models.smpo.SMPO_initialize(
                L=15,
                initializer=tn4ml.initializers.randn(1e-9, dtype=jnp.float64),
                key=jax.random.key(42),
                shape_method="even",
                spacing=5,
                bond_dim=4,
                phys_dim=(2, 2),
                cyclic=False,
            )
        ),
    ],
)
def test_LogReLUFrobNorm_SMPO(model):  # noqa: N802
    """Test LogReLUFrobNorm SMPO."""
    loss = tn4ml.metrics.LogReLUFrobNorm(model)
    print(jax.device_get(loss))
    assert jax.device_get(loss) >= 0.0


@pytest.mark.parametrize(
    "model",
    [
        (
            tn4ml.models.mps.MPS_initialize(
                L=10,
                initializer=randn(1e-1),
                key=jax.random.key(42),
                shape_method="even",
                bond_dim=4,
                phys_dim=2,
                cyclic=False,
            )
        ),
        # (tn4ml.models.mps.MPS_initialize(L=20, initializer=tn4ml.initializers.randn(1e-2, dtype=jnp.float64),
        #                                 key=jax.random.key(42), shape_method='noteven',
        #                                 bond_dim=4, phys_dim=2, cyclic=False))
    ],
)
def test_LogReLUFrobNorm_TrainableMPS(model):  # noqa: N802
    """Test LogReLUFrobNorm TrainableMPS."""
    loss = tn4ml.metrics.LogReLUFrobNorm(model)
    assert jax.device_get(loss) >= 0.0


@pytest.mark.parametrize(
    "model",
    [
        (
            tn4ml.models.mpo.MPO_initialize(
                L=10,
                initializer=randn(1e-1),
                key=jax.random.key(42),
                shape_method="even",
                bond_dim=4,
                phys_dim=(2, 2),
                cyclic=False,
            )
        ),
        # (tn4ml.models.mpo.MPO_initialize(L=20, initializer=tn4ml.initializers.randn(1e-2, dtype=jnp.float64),
        #                                 key=jax.random.key(42), shape_method='noteven',
        #                                 bond_dim=4, phys_dim=(2,2), cyclic=False))
    ],
)
def test_LogReLUFrobNorm_TrainableMPO(model):  # noqa: N802
    """Test LogReLUFrobNorm TrainableMPO."""
    loss = tn4ml.metrics.LogReLUFrobNorm(model)
    assert jax.device_get(loss) >= 0.0


# TODO add more MPO initializations from quimb
@pytest.mark.parametrize(
    "model",
    [
        (qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)),
        (qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2, cyclic=True)),
    ],
)
def test_reg_norm_quad_MPS(model):  # noqa: N802
    """Test reg norm quad MPS."""
    loss = tn4ml.metrics.QuadFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)


# TODO add more MPO initializations from quimb
@pytest.mark.parametrize(
    "model",
    [
        (qtn.MPO_rand(10, bond_dim=2, phys_dim=2)),
        (qtn.MPO_rand(10, bond_dim=2, phys_dim=2, cyclic=True)),
    ],
)
def test_reg_norm_quad_MPO(model):  # noqa: N802
    """Test reg norm quad MPO."""
    loss = tn4ml.metrics.QuadFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)


def test_reg_norm_quad_SMPO():  # noqa: N802
    """Test reg norm quad SMPO."""
    model = tn4ml.models.smpo.SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        cyclic=False,
    )
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)


def test_reg_norm_quad_TrainableMPS():  # noqa: N802
    """Test reg norm quad TrainableMPS."""
    model = tn4ml.models.mps.MPS_initialize(
        L=10,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        bond_dim=4,
        phys_dim=2,
        cyclic=False,
    )
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)


def test_reg_norm_quad_TrainableMPO():  # noqa: N802
    """Test reg norm quad TrainableMPO."""
    model = tn4ml.models.mpo.MPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        bond_dim=4,
        phys_dim=(2, 2),
        cyclic=False,
    )
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)


def test_LogQuadNorm_SMPO_with_MPS_rand_state():  # noqa: N802
    """Test LogQuadNorm SMPO with MPS rand state."""
    model = smpo.SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        cyclic=False,
    )
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    error = metrics.LogQuadNorm(model, data)
    assert isinstance(jax.device_get(error), np.ndarray)


def test_LogQuadNorm_SMPO_with_embedded_numpy_array():  # noqa: N802
    """Test LogQuadNorm SMPO with embedded numpy array."""
    model = smpo.SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        cyclic=False,
    )
    data = np.random.rand(10)  # noqa: NPY002
    embedded_data = embeddings.embed(data, phi=embeddings.TrigonometricEmbedding())
    error = metrics.LogQuadNorm(model, embedded_data)
    assert isinstance(jax.device_get(error), np.ndarray)


def test_error_quad_SMPO_with_MPS_rand_state():  # noqa: N802
    """Test error quad SMPO with MPS rand state."""
    model = smpo.SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        cyclic=False,
    )
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    error = metrics.QuadNorm(model, data)
    assert isinstance(jax.device_get(error), np.ndarray)


def test_error_quad_SMPO_with_embedded_numpy_array():  # noqa: N802
    """Test error quad SMPO with embedded numpy array."""
    model = smpo.SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        cyclic=False,
    )
    data = np.random.rand(10)  # noqa: NPY002
    embedded_data = embeddings.embed(data, phi=embeddings.TrigonometricEmbedding())
    error = metrics.QuadNorm(model, embedded_data)
    assert isinstance(jax.device_get(error), np.ndarray)


def test_CrossEntropySoftmax():  # noqa: N802
    """Test CrossEntropySoftmax."""
    model = tn4ml.models.smpo.SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        spacing=10,
        bond_dim=4,
        phys_dim=(2, 3),
        cyclic=False,
    )
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    targets = jnp.array([0, 1, 0])
    loss = tn4ml.metrics.CrossEntropySoftmax(model, data, targets)
    assert isinstance(loss, jnp.ndarray)


def test_CrossEntropySoftmax_with_embedded_numpy_array():  # noqa: N802
    """Test CrossEntropySoftmax with embedded numpy array."""
    model = tn4ml.models.smpo.SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        spacing=10,
        bond_dim=4,
        phys_dim=(2, 3),
        cyclic=False,
    )
    data = np.random.rand(10)  # noqa: NPY002
    embedded_data = embeddings.embed(data, phi=embeddings.TrigonometricEmbedding())
    targets = jnp.array([0, 1, 0])
    loss = tn4ml.metrics.CrossEntropySoftmax(model, embedded_data, targets)
    assert isinstance(loss, jnp.ndarray)


# --- NoReg ---


def test_NoReg():  # noqa: N802
    """Test NoReg."""
    result = metrics.NoReg(42)
    assert result == 0


def test_NoReg_with_model():  # noqa: N802
    """Test NoReg with model."""
    model = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    result = metrics.NoReg(model)
    assert result == 0


# --- LogPowFrobNorm ---


@pytest.mark.parametrize(
    "model",
    [
        qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2),
        qtn.MPO_rand(10, bond_dim=2, phys_dim=2),
    ],
)
def test_LogPowFrobNorm(model):  # noqa: N802
    """Test LogPowFrobNorm."""
    loss = metrics.LogPowFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)


# --- Softmax ---


def test_Softmax_basic():  # noqa: N802
    """Test Softmax basic."""
    z = jnp.array([1.0, 2.0, 3.0])
    result = metrics.Softmax(z, 2)
    assert isinstance(float(result), float)
    assert 0.0 <= float(result) <= 1.0


def test_Softmax_sums_to_one():  # noqa: N802
    """Test Softmax sums to one."""
    z = jnp.array([1.0, 2.0, 3.0])
    total = sum(float(metrics.Softmax(z, i)) for i in range(3))
    assert total == pytest.approx(1.0)


# --- MeanSquaredError ---


def test_MeanSquaredError():  # noqa: N802
    """Test MeanSquaredError."""
    model = tn4ml.models.smpo.SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        spacing=10,
        bond_dim=4,
        phys_dim=(2, 3),
        cyclic=False,
    )
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    targets = jnp.array([0.5, 0.3, 0.2])
    loss = metrics.MeanSquaredError(model, data, targets)
    assert float(loss) >= 0.0


# --- SemiSupervisedLoss ---


def test_SemiSupervisedLoss():  # noqa: N802
    """Test SemiSupervisedLoss."""
    model = tn4ml.models.smpo.SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=jax.random.key(42),
        shape_method="even",
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        cyclic=False,
    )
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    loss = metrics.SemiSupervisedLoss(model, data, y_true=0.5)
    assert isinstance(float(loss), float)


# --- Error-path branches (ValueError when model has more tensors than data) ---


def test_NegLogLikelihood_raises_when_model_larger():  # noqa: N802
    """Test NegLogLikelihood raises when model larger."""
    model = qtn.MPS_rand_state(12, bond_dim=2, phys_dim=2)
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    with pytest.raises(ValueError, match="higher or equal"):
        metrics.NegLogLikelihood(model, data)


def test_CrossEntropySoftmax_raises_when_model_larger():  # noqa: N802
    """Test CrossEntropySoftmax raises when model larger."""
    model = _smpo(L=12, spacing=12, phys_dim=(2, 3))
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    with pytest.raises(ValueError, match="higher or equal"):
        metrics.CrossEntropySoftmax(model, data, jnp.array([0, 1, 0]))


# --- TransformedSquaredNorm: model with fewer tensors than data ---


def test_TransformedSquaredNorm_model_smaller_than_data():  # noqa: N802
    """Test TransformedSquaredNorm model smaller than data."""
    model = _smpo(L=10, spacing=2, phys_dim=(2, 2))
    data = qtn.MPS_rand_state(12, bond_dim=2, phys_dim=2)
    loss = metrics.TransformedSquaredNorm(model, data)
    assert jax.device_get(loss) >= 0.0


# --- LogPowFrobNorm / QuadFrobNorm: SMPO branch ---


def test_LogPowFrobNorm_SMPO():  # noqa: N802
    """Test LogPowFrobNorm SMPO."""
    loss = metrics.LogPowFrobNorm(_smpo())
    assert isinstance(jax.device_get(loss), np.ndarray)


def test_QuadFrobNorm_SMPO():  # noqa: N802
    """Test QuadFrobNorm SMPO."""
    loss = metrics.QuadFrobNorm(_smpo())
    assert isinstance(jax.device_get(loss), np.ndarray)


# --- SemiSupervisedNLL ---


def test_SemiSupervisedNLL():  # noqa: N802
    """Test SemiSupervisedNLL."""
    model = _smpo(L=10, spacing=10, phys_dim=(2, 2))
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    loss = metrics.SemiSupervisedNLL(model, data, y_true=jnp.array([1]))
    assert jax.device_get(loss) is not None


# --- OptaxWrapper ---


def test_OptaxWrapper_requires_loss():  # noqa: N802
    """Test OptaxWrapper requires loss."""
    with pytest.raises(AssertionError):
        metrics.OptaxWrapper()


def test_OptaxWrapper_MPS_unsupervised():  # noqa: N802
    """Test OptaxWrapper MPS unsupervised."""
    model = _mps(L=10, phys_dim=2)
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    loss_fn = metrics.OptaxWrapper(optax.squared_error)
    value = loss_fn(model, data)
    assert jax.device_get(value) is not None


def test_OptaxWrapper_SMPO_unsupervised():  # noqa: N802
    """Test OptaxWrapper SMPO unsupervised."""
    model = _smpo(L=10, spacing=2, phys_dim=(2, 2))
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    loss_fn = metrics.OptaxWrapper(optax.squared_error)
    value = loss_fn(model, data)
    assert jax.device_get(value) is not None


def test_OptaxWrapper_SMPO_supervised():  # noqa: N802
    """Test OptaxWrapper SMPO supervised."""
    model = _smpo(L=10, spacing=10, phys_dim=(2, 3))
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    loss_fn = metrics.OptaxWrapper(optax.softmax_cross_entropy)
    value = loss_fn(model, data, y_true=jnp.array([0.0, 1.0, 0.0]))
    assert jax.device_get(value) is not None


# --- CrossEntropyWeighted ---


def test_CrossEntropyWeighted():  # noqa: N802
    """Test CrossEntropyWeighted."""
    model = _smpo(L=10, spacing=10, phys_dim=(2, 3))
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    loss_fn = metrics.CrossEntropyWeighted(class_weights=jnp.array([1.0, 2.0, 1.0]))
    value = loss_fn(model, data, y_true=jnp.array([0.0, 1.0, 0.0]))
    assert jax.device_get(value) is not None


# --- CombinedLoss ---


def test_CombinedLoss_raises_without_data():  # noqa: N802
    """Test CombinedLoss raises without data."""
    with pytest.raises(ValueError, match="Provide input data"):
        metrics.CombinedLoss(_smpo(), None)


def test_CombinedLoss_unsupervised():  # noqa: N802
    """Test CombinedLoss unsupervised."""
    model = _smpo(L=10, spacing=2, phys_dim=(2, 2))
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    loss = metrics.CombinedLoss(
        model, data, error=metrics.LogQuadNorm, reg=metrics.NoReg
    )
    assert jax.device_get(loss) is not None


def test_CombinedLoss_unsupervised_with_numpy_and_embedding():  # noqa: N802
    """Test CombinedLoss unsupervised with numpy and embedding."""
    model = _smpo(L=10, spacing=2, phys_dim=(2, 2))
    data = np.random.rand(4, 10)  # noqa: NPY002
    loss = metrics.CombinedLoss(
        model,
        data,
        error=metrics.LogQuadNorm,
        reg=metrics.LogReLUFrobNorm,
        embedding=embeddings.TrigonometricEmbedding(),
    )
    assert jax.device_get(loss) is not None


def test_CombinedLoss_numpy_without_embedding_raises():  # noqa: N802
    """Test CombinedLoss numpy without embedding raises."""
    model = _smpo(L=10, spacing=2, phys_dim=(2, 2))
    data = np.random.rand(4, 10)  # noqa: NPY002
    with pytest.raises(ValueError, match="embedding"):
        metrics.CombinedLoss(model, data, error=metrics.LogQuadNorm)


def test_CombinedLoss_supervised():  # noqa: N802
    """Test CombinedLoss supervised."""
    model = _smpo(L=10, spacing=10, phys_dim=(2, 3))
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    loss = metrics.CombinedLoss(
        model,
        data,
        y_true=jnp.array([0, 1, 0]),
        error=metrics.CrossEntropySoftmax,
        reg=metrics.NoReg,
    )
    assert jax.device_get(loss) is not None
