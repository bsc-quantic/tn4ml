"""Test TensorNetwork model class."""

import jax
import jax.numpy as jnp
import pytest
import quimb.tensor as qtn

from tn4ml.initializers import randn
from tn4ml.models.tn import TensorNetwork, TN_initialize, trainable_wrapper

jax.config.update("jax_enable_x64", True)


# --- TN_initialize ---


@pytest.mark.parametrize("cyclic", [False, True])
def test_TN_initialize_from_shapes(cyclic):  # noqa: N802
    key = jax.random.PRNGKey(42)
    # Boundary tensors are 2D (no dangling bond on that side), middle is 3D
    shapes = [(3, 2), (3, 3, 2), (3, 2)]
    inds = [["bond_0", "k0"], ["bond_0", "bond_1", "k1"], ["bond_1", "k2"]]
    if cyclic:
        shapes = [(3, 3, 2), (3, 3, 2), (3, 3, 2)]
        inds = [
            ["bond_2", "bond_0", "k0"],
            ["bond_0", "bond_1", "k1"],
            ["bond_1", "bond_2", "k2"],
        ]
    tn = TN_initialize(
        shapes=shapes, key=key, initializer=randn(1e-1), inds=inds, cyclic=cyclic
    )
    assert tn.norm() == pytest.approx(1.0, abs=1e-5)
    assert len(tn.tensors) == 3


def test_TN_initialize_from_arrays():  # noqa: N802
    arrays = [jnp.ones((3, 2)), jnp.ones((3, 3, 2)), jnp.ones((3, 2))]
    inds = [["bond_0", "k0"], ["bond_0", "bond_1", "k1"], ["bond_1", "k2"]]
    tn = TN_initialize(arrays=arrays, inds=inds)
    assert len(tn.tensors) == 3
    assert tn.norm() == pytest.approx(1.0, abs=1e-5)


def test_TN_initialize_no_arrays_no_shapes():  # noqa: N802
    with pytest.raises(ValueError, match="Provide either"):
        TN_initialize()


def test_TN_initialize_no_inds():  # noqa: N802
    with pytest.raises(ValueError, match="Provide indices"):
        TN_initialize(shapes=[(1, 3, 2)], key=jax.random.PRNGKey(0))


def test_TN_initialize_mismatched_arrays_inds():  # noqa: N802
    arrays = [jnp.ones((1, 3, 2))]
    inds = [["bond_0", "k0"], ["bond_0", "bond_1", "k1"]]
    with pytest.raises(ValueError, match="same"):
        TN_initialize(arrays=arrays, inds=inds)


def test_TN_initialize_mismatched_shapes_inds():  # noqa: N802
    shapes = [(1, 3, 2)]
    inds = [["bond_0", "k0"], ["bond_0", "bond_1", "k1"]]
    with pytest.raises(ValueError, match="same"):
        TN_initialize(shapes=shapes, key=jax.random.PRNGKey(0), inds=inds)


# --- TensorNetwork methods ---


def _make_tn():
    key = jax.random.PRNGKey(42)
    shapes = [(3, 2), (3, 3, 2), (3, 2)]
    inds = [["bond_0", "k0"], ["bond_0", "bond_1", "k1"], ["bond_1", "k2"]]
    return TN_initialize(shapes=shapes, key=key, initializer=randn(1e-1), inds=inds)


def test_tn_copy():
    tn = _make_tn()
    tn_copy = tn.copy()
    assert len(tn_copy.tensors) == len(tn.tensors)
    assert tn_copy.norm() == pytest.approx(tn.norm(), abs=1e-5)


def test_tn_deep_copy():
    tn = _make_tn()
    tn_copy = tn.copy(deep=True)
    assert len(tn_copy.tensors) == len(tn.tensors)


def test_TN_initialize_from_shapes_no_initializer():  # noqa: N802
    # No initializer -> falls back to the numpy RNG path.
    shapes = [(3, 2), (3, 3, 2), (3, 2)]
    inds = [["bond_0", "k0"], ["bond_0", "bond_1", "k1"], ["bond_1", "k2"]]
    tn = TN_initialize(shapes=shapes, inds=inds)
    assert len(tn.tensors) == 3
    assert tn.norm() == pytest.approx(1.0, abs=1e-5)


def test_tn_canonize():
    tn = _make_tn()
    tn.canonize(1)
    assert len(tn.tensors) == 3


def test_tn_norm():
    tn = _make_tn()
    assert isinstance(float(tn.norm()), float)


def test_tn_normalize():
    key = jax.random.PRNGKey(0)
    shapes = [(3, 2), (3, 3, 2), (3, 2)]
    inds = [["bond_0", "k0"], ["bond_0", "bond_1", "k1"], ["bond_1", "k2"]]
    tensors = []
    for i, shape in enumerate(shapes):
        array = jax.random.normal(key, shape) * 5.0
        tensors.append(qtn.Tensor(array, inds=inds[i], tags=f"I{i}"))
    tn = TensorNetwork(tensors)
    tn.normalize()
    assert tn.norm() == pytest.approx(1.0, abs=1e-4)


def test_tn_normalize_with_insert():
    tn = _make_tn()
    # Scale up
    for t in tn.tensors:
        t.modify(data=t.data * 3.0)
    tn.normalize(insert=0)
    assert tn.norm() == pytest.approx(1.0, abs=1e-4)


def test_tn_normalize_invalid_insert():
    tn = _make_tn()
    with pytest.raises(IndexError, match="out of bounds"):
        tn.normalize(insert=100)


# --- trainable_wrapper ---


def test_trainable_wrapper():
    mps = qtn.MPS_rand_state(5, bond_dim=2, phys_dim=2)
    tn = trainable_wrapper(mps)
    assert isinstance(tn, TensorNetwork)
    assert len(tn.tensors) == 5
