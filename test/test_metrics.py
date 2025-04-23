import pytest
import tn4ml
from numbers import Number
import numpy as np
import jax.numpy as jnp
import jax
import optax
import numpy as np
import quimb.tensor as qtn
import tn4ml.metrics as metrics
import tn4ml.models.smpo as smpo
import tn4ml.embeddings as embeddings
import quimb.tensor as qtn

@pytest.mark.parametrize("model,data", [(qtn.MPS_rand_state(20, bond_dim=2, phys_dim=2), qtn.MPS_rand_state(20, bond_dim=5, phys_dim=2))])
def test_NegLogLikelihood_1(model, data):
    loss = tn4ml.metrics.NegLogLikelihood(model, data)
    assert loss >= 0.0
    # check if physical dimensions match 
    assert model.tensors[0].shape[-1] == data.tensors[0].shape[-1]
    assert len(model.tensors) == len(data.tensors)

@pytest.mark.parametrize("model,data", [(qtn.MPS_rand_state(20, bond_dim=2, phys_dim=2, cyclic=True), qtn.MPS_rand_state(20, bond_dim=5, phys_dim=2))])
def test_NegLogLikelihood_2(model, data):
    loss = tn4ml.metrics.NegLogLikelihood(model, data)
    assert loss >= 0.0
    # check if physical dimensions match 
    assert model.tensors[0].shape[-1] == data.tensors[0].shape[-1]
    assert len(model.tensors) == len(data.tensors)

@pytest.mark.parametrize("model,data", [(qtn.MPS_rand_state(20, bond_dim=2, phys_dim=2, cyclic=True), qtn.MPS_rand_state(20, bond_dim=5, phys_dim=2, cyclic=True))])
def test_NegLogLikelihood_3(model, data):
    loss = tn4ml.metrics.NegLogLikelihood(model, data)
    assert loss >= 0.0
    # check if physical dimensions match 
    assert model.tensors[0].shape[-1] == data.tensors[0].shape[-1]
    assert len(model.tensors) == len(data.tensors)

@pytest.mark.parametrize("model,data", [(qtn.MPS_rand_state(10, bond_dim=2, phys_dim=3), np.random.rand(10,))])
def test_NegLogLikelihood_4(model,data):
    embedding = tn4ml.embeddings.FourierEmbedding(p=3)
    phi = tn4ml.embeddings.embed(data, phi=embedding)
    loss = tn4ml.metrics.NegLogLikelihood(model, phi)
    assert loss >= 0.0
    # check if physical dimensions match 
    assert model.tensors[0].shape[-1] == phi.tensors[0].shape[-1]
    assert len(model.tensors) == len(phi.tensors)

# TODO - when SMPO is fully working!
@pytest.mark.parametrize("model,data", [(tn4ml.models.smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                                                            key=jax.random.key(42), shape_method='even',
                                                                            spacing=2, bond_dim=4,
                                                                            phys_dim=(2,2), cyclic=False),
                                        np.random.rand(10,))])
def test_transformed_squared_norm(model,data):
    embedding = tn4ml.embeddings.TrigonometricEmbedding()
    phi = tn4ml.embeddings.embed(data, phi=embedding)
    loss = tn4ml.metrics.TransformedSquaredNorm(model, phi)
    assert loss >= 0.0
    # check if physical dimensions match 
    assert model.tensors[0].shape[-1] == phi.tensors[0].shape[-1]
    assert len(model.tensors) == len(phi.tensors)

@pytest.mark.parametrize("model,data", [(tn4ml.models.smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                                                            key=jax.random.key(42), shape_method='even',
                                                                            spacing=2, bond_dim=4,
                                                                            phys_dim=(2,2), cyclic=False),
                                        qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2))])
def test_transformed_squared_norm_2(model,data):
    loss = tn4ml.metrics.TransformedSquaredNorm(model, data)
    assert loss >= 0.0
    # check if physical dimensions match 
    assert model.tensors[0].shape[-1] == data.tensors[0].shape[-1]
    assert len(model.tensors) == len(data.tensors)

# TODO add more MPO initializations from quimb   
@pytest.mark.parametrize("model", [
        (qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)),
        (qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2, cyclic=True))])
def test_LogFrobNorm_MPS(model):
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)

# TODO add more MPO initializations from quimb
@pytest.mark.parametrize("model", [
    (qtn.MPO_rand(10, bond_dim=2, phys_dim=2)),
    (qtn.MPO_rand(10, bond_dim=2, phys_dim=2, cyclic=True))])
def test_LogFrobNorm_MPO(model):
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)

@pytest.mark.parametrize("model", [
    (tn4ml.models.smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                    key=jax.random.key(42), shape_method='even',
                                    spacing=2, bond_dim=4,
                                    phys_dim=(2,2), cyclic=False)),
    (tn4ml.models.smpo.SMPO_initialize(L=15, initializer=tn4ml.initializers.randn(1e-9, dtype=jnp.float64),
                                    key=jax.random.key(42), shape_method='even',
                                    spacing=5, bond_dim=4,
                                    phys_dim=(2,2), cyclic=False))])
def test_LogFrobNorm_SMPO(model):
    loss = tn4ml.metrics.LogFrobNorm(model)
    print(jax.device_get(loss))
    assert isinstance(jax.device_get(loss), np.ndarray)

@pytest.mark.parametrize("model", [
    (tn4ml.models.mps.MPS_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                    key=jax.random.key(42), shape_method='even',
                                    bond_dim=4, phys_dim=2, cyclic=False)),
    (tn4ml.models.mps.MPS_initialize(L=20, initializer=tn4ml.initializers.randn(1e-9, dtype=jnp.float64),
                                    key=jax.random.key(42), shape_method='noteven',
                                    bond_dim=4, phys_dim=2, cyclic=False))])
def test_LogFrobNorm_TrainableMPS(model):
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)

@pytest.mark.parametrize("model", [
    (tn4ml.models.mpo.MPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                    key=jax.random.key(42), shape_method='even',
                                    bond_dim=4, phys_dim=(2,2), cyclic=False)),
    (tn4ml.models.mpo.MPO_initialize(L=20, initializer=tn4ml.initializers.randn(1e-9, dtype=jnp.float64),
                                    key=jax.random.key(42), shape_method='noteven',
                                    bond_dim=4, phys_dim=(2,2), cyclic=False))])
def test_LogFrobNorm_TrainableMP0(model):
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)
    
# TODO add more MPS initializations from quimb   
@pytest.mark.parametrize("model", [
        (qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)),
        (qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2, cyclic=True))])
def test_LogReLUFrobNorm_MPS(model):
    loss = tn4ml.metrics.LogReLUFrobNorm(model)
    assert jax.device_get(loss) >= 0.0

# TODO add more MPO initializations from quimb
@pytest.mark.parametrize("model", [
    (qtn.MPO_rand(10, bond_dim=2, phys_dim=2)),
    (qtn.MPO_rand(10, bond_dim=2, phys_dim=2, cyclic=True))])
def test_LogReLUFrobNorm_MPO(model):
    loss = tn4ml.metrics.LogReLUFrobNorm(model)
    assert jax.device_get(loss) >= 0.0

@pytest.mark.parametrize("model", [
    (tn4ml.models.smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                    key=jax.random.key(42), shape_method='even',
                                    spacing=2, bond_dim=4,
                                    phys_dim=(2,2), cyclic=False)),
    (tn4ml.models.smpo.SMPO_initialize(L=15, initializer=tn4ml.initializers.randn(1e-9, dtype=jnp.float64),
                                    key=jax.random.key(42), shape_method='even',
                                    spacing=5, bond_dim=4,
                                    phys_dim=(2,2), cyclic=False))])
def test_LogReLUFrobNorm_SMPO(model):
    loss = tn4ml.metrics.LogReLUFrobNorm(model)
    print(jax.device_get(loss))
    assert jax.device_get(loss) >= 0.0

@pytest.mark.parametrize("model", [
    (tn4ml.models.mps.MPS_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                    key=jax.random.key(42), shape_method='even',
                                    bond_dim=4, phys_dim=2, cyclic=False)),
    # (tn4ml.models.mps.MPS_initialize(L=20, initializer=tn4ml.initializers.randn(1e-2, dtype=jnp.float64),
    #                                 key=jax.random.key(42), shape_method='noteven',
    #                                 bond_dim=4, phys_dim=2, cyclic=False))
                                    ])
def test_LogReLUFrobNorm_TrainableMPS(model):
    loss = tn4ml.metrics.LogReLUFrobNorm(model)
    assert jax.device_get(loss) >= 0.0

@pytest.mark.parametrize("model", [
    (tn4ml.models.mpo.MPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                    key=jax.random.key(42), shape_method='even',
                                    bond_dim=4, phys_dim=(2,2), cyclic=False)),
    # (tn4ml.models.mpo.MPO_initialize(L=20, initializer=tn4ml.initializers.randn(1e-2, dtype=jnp.float64),
    #                                 key=jax.random.key(42), shape_method='noteven',
    #                                 bond_dim=4, phys_dim=(2,2), cyclic=False))
    ])
def test_LogReLUFrobNorm_TrainableMPO(model):
    loss = tn4ml.metrics.LogReLUFrobNorm(model)
    assert jax.device_get(loss) >= 0.0
    
# TODO add more MPO initializations from quimb   
@pytest.mark.parametrize("model", [
        (qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)),
        (qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2, cyclic=True))])
def test_reg_norm_quad_MPS(model):
    loss = tn4ml.metrics.QuadFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)

# TODO add more MPO initializations from quimb
@pytest.mark.parametrize("model", [
    (qtn.MPO_rand(10, bond_dim=2, phys_dim=2)),
    (qtn.MPO_rand(10, bond_dim=2, phys_dim=2, cyclic=True))])
def test_reg_norm_quad_MPO(model):
    loss = tn4ml.metrics.QuadFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)

def test_reg_norm_quad_SMPO():
    model = tn4ml.models.smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                              key=jax.random.key(42), shape_method='even',
                                              spacing=2, bond_dim=4,
                                              phys_dim=(2,2), cyclic=False)
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)

def test_reg_norm_quad_TrainableMPS():
    model = tn4ml.models.mps.MPS_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                            key=jax.random.key(42), shape_method='even',
                                            bond_dim=4, phys_dim=2, cyclic=False)
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)

def test_reg_norm_quad_TrainableMPO():
    model = tn4ml.models.mpo.MPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                            key=jax.random.key(42), shape_method='even',
                                            bond_dim=4, phys_dim=(2,2), cyclic=False)
    loss = tn4ml.metrics.LogFrobNorm(model)
    assert isinstance(jax.device_get(loss), np.ndarray)

def test_LogQuadNorm_SMPO_with_MPS_rand_state():
    model = smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                 key=jax.random.key(42), shape_method='even',
                                 spacing=2, bond_dim=4,
                                 phys_dim=(2,2), cyclic=False)
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    error = metrics.LogQuadNorm(model, data)
    assert isinstance(jax.device_get(error), np.ndarray)

def test_LogQuadNorm_SMPO_with_embedded_numpy_array():
    model = smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                 key=jax.random.key(42), shape_method='even',
                                 spacing=2, bond_dim=4,
                                 phys_dim=(2,2), cyclic=False)
    data = np.random.rand(10)
    embedded_data = embeddings.embed(data, phi=embeddings.TrigonometricEmbedding())
    error = metrics.LogQuadNorm(model, embedded_data)
    assert isinstance(jax.device_get(error), np.ndarray)

def test_error_quad_SMPO_with_MPS_rand_state():
    model = smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                 key=jax.random.key(42), shape_method='even',
                                 spacing=2, bond_dim=4,
                                 phys_dim=(2,2), cyclic=False)
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    error = metrics.QuadNorm(model, data)
    assert isinstance(jax.device_get(error), np.ndarray)

def test_error_quad_SMPO_with_embedded_numpy_array():
    model = smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                 key=jax.random.key(42), shape_method='even',
                                 spacing=2, bond_dim=4,
                                 phys_dim=(2,2), cyclic=False)
    data = np.random.rand(10)
    embedded_data = embeddings.embed(data, phi=embeddings.TrigonometricEmbedding())
    error = metrics.QuadNorm(model, embedded_data)
    assert isinstance(jax.device_get(error), np.ndarray)

def test_CrossEntropySoftmax():
    model = tn4ml.models.smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                              key=jax.random.key(42), shape_method='even',
                                              spacing=10, bond_dim=4,
                                              phys_dim=(2,3), cyclic=False)
    data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    targets = jnp.array([0, 1, 0])
    loss = tn4ml.metrics.CrossEntropySoftmax(model, data, targets)
    assert isinstance(loss, jnp.ndarray)

def test_CrossEntropySoftmax_with_embedded_numpy_array():
    model = tn4ml.models.smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
                                              key=jax.random.key(42), shape_method='even',
                                              spacing=10, bond_dim=4,
                                              phys_dim=(2,3), cyclic=False)
    data = np.random.rand(10)
    embedded_data = embeddings.embed(data, phi=embeddings.TrigonometricEmbedding())
    targets = jnp.array([0, 1, 0])
    loss = tn4ml.metrics.CrossEntropySoftmax(model, embedded_data, targets)
    assert isinstance(loss, jnp.ndarray)

# def test_OptaxWrapper():
#     # Test for SMPO with numpy data
#     model = tn4ml.models.smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
#                                                 key=jax.random.key(42), shape_method='even',
#                                                 spacing=2, bond_dim=4,
#                                                 phys_dim=(2,2), cyclic=False)
#     data = np.random.rand(10)
#     embedded_data = embeddings.embed(data, phi=embeddings.TrigonometricEmbedding())
#     loss_fn = optax.squared_error
#     loss = tn4ml.metrics.OptaxWrapper(loss_fn)
#     loss_value = loss(model, embedded_data).mean()
#     assert isinstance(jax.device_get(loss_value), Number)

#     # Test for SMPO with qtn.MPS_rand_state data
#     model = tn4ml.models.smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
#                                                 key=jax.random.key(42), shape_method='even',
#                                                 spacing=2, bond_dim=4,
#                                                 phys_dim=(2,2), cyclic=False)
#     data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
#     loss_fn = optax.squared_error
#     loss = tn4ml.metrics.OptaxWrapper(loss_fn)
#     loss_value = loss(model, data).mean()
#     assert isinstance(jax.device_get(loss_value), Number)

#     # Test for TrainableMPS with numpy data
#     model = tn4ml.models.mps.MPS_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
#                                             key=jax.random.key(42), shape_method='even',
#                                             bond_dim=4, phys_dim=2, cyclic=False)
#     data = np.random.rand(10)
#     embedded_data = embeddings.embed(data, phi=embeddings.TrigonometricEmbedding())
#     loss_fn = optax.softmax_cross_entropy
#     loss = tn4ml.metrics.OptaxWrapper(loss_fn)
#     loss_value = loss(model, embedded_data).mean()
#     assert isinstance(jax.device_get(loss_value)[0], Number)

#     # Test for TrainableMPS with qtn.MPS_rand_state data
#     model = tn4ml.models.mps.MPS_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
#                                             key=jax.random.key(42), shape_method='even',
#                                             bond_dim=4, phys_dim=2, cyclic=False)
#     data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
#     loss_fn = optax.softmax_cross_entropy
#     loss = tn4ml.metrics.OptaxWrapper(loss_fn)
#     loss_value = loss(model, data).mean()
#     assert isinstance(jax.device_get(loss_value), Number)

#     # Test for SMPO with numpy data and supervised loss
#     model = tn4ml.models.smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
#                                                 key=jax.random.key(42), shape_method='even',
#                                                 spacing=10, bond_dim=4,
#                                                 phys_dim=(2,10), cyclic=False)
#     data = np.random.rand(10)
#     embedded_data = embeddings.embed(data, phi=embeddings.TrigonometricEmbedding())
#     targets = np.random.randint(0, 2, size=(10,))
#     loss_fn = optax.softmax_cross_entropy
#     loss = tn4ml.metrics.OptaxWrapper(loss_fn)
#     loss_value = loss(model, embedded_data, targets=targets).mean()
#     assert isinstance(jax.device_get(loss_value), Number)

#     # Test for SMPO with qtn.MPS_rand_state data and supervised loss
#     model = tn4ml.models.smpo.SMPO_initialize(L=10, initializer=jax.nn.initializers.orthogonal(),
#                                                 key=jax.random.key(42), shape_method='even',
#                                                 spacing=10, bond_dim=4,
#                                                 phys_dim=(2,10), cyclic=False)
#     data = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
#     targets = np.random.randint(0, 2, size=(10,))
#     loss_fn = optax.softmax_cross_entropy
#     loss = tn4ml.metrics.OptaxWrapper(loss_fn)
#     loss_value = loss(model, data, targets=targets).mean()
#     assert isinstance(jax.device_get(loss_value), Number)

# TODO add tests for CombinedLoss: SMPO, TrainableMPS, TrainableMPO