"""Test TrainableMPS initialization."""

import jax
import pytest
import quimb.tensor as qtn
from jax.nn.initializers import he_normal, normal, orthogonal, uniform

from tn4ml.initializers import gramschmidt, rand_unitary, randn
from tn4ml.models.mps import MPS_initialize, trainable_wrapper

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("L", "initializer", "shape_method", "bond_dim", "phys_dim", "cyclic"),
    [
        (10, orthogonal(), "even", 5, 2, False),
        (10, he_normal(), "even", 5, 2, False),
        (10, normal(1e-2), "even", 5, 2, False),
        (10, uniform(), "even", 5, 2, False),
        (10, gramschmidt("normal", 1e-3), "even", 5, 2, False),
        (10, gramschmidt("uniform", 1e-3), "even", 5, 2, False),
        (10, gramschmidt("normal", 1e-3), "noteven", 10, 20, False),
        (10, gramschmidt("uniform", 1e-3), "noteven", 10, 20, False),
        (10, randn(1e-7), "noteven", 10, 2, False),
        (10, randn(0.5), "noteven", 10, 2, False),
        (10, gramschmidt("normal", 1e-3), "even", 5, 2, True),
        (10, gramschmidt("uniform", 1e-3), "even", 5, 2, True),
        (10, rand_unitary(), "even", 5, 2, True),
    ],
)
def test_MPS_initialize(L, initializer, shape_method, bond_dim, phys_dim, cyclic):  # noqa: N802
    key = jax.random.PRNGKey(42)
    mps = MPS_initialize(
        L,
        initializer=initializer,
        key=key,
        shape_method=shape_method,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        cyclic=cyclic,
    )
    assert mps.norm() == pytest.approx(1.0)


@pytest.mark.parametrize("mps", [qtn.MPS_rand_state(20, bond_dim=2, phys_dim=2)])
def test_trainable_wrapper(mps):
    mps = trainable_wrapper(mps)
    assert mps.norm() == pytest.approx(1.0)


# --- Classification MPS (class_index / class_dim) ---


@pytest.mark.parametrize("shape_method", ["even", "noteven"])
@pytest.mark.parametrize("add_identity", [False, True])
def test_MPS_initialize_classification(shape_method, add_identity):  # noqa: N802
    key = jax.random.PRNGKey(42)
    mps = MPS_initialize(
        L=8,
        initializer=randn(1e-1),
        key=key,
        shape_method=shape_method,
        bond_dim=4,
        phys_dim=2,
        class_index=4,
        class_dim=3,
        add_identity=add_identity,
        add_to_output=add_identity,
    )
    # Output node carries the extra class dimension of size 3.
    output_tensor = mps.tensors[3]
    assert 3 in output_tensor.shape


def test_MPS_initialize_classification_output_at_end():  # noqa: N802
    key = jax.random.PRNGKey(0)
    mps = MPS_initialize(
        L=6,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=4,
        phys_dim=2,
        class_index=6,
        class_dim=2,
    )
    assert 2 in mps.tensors[5].shape


# --- Optional regression features: insert, compress, canonical_center ---


def test_MPS_initialize_insert():  # noqa: N802
    key = jax.random.PRNGKey(1)
    mps = MPS_initialize(
        L=8,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=4,
        phys_dim=2,
        insert=2,
    )
    assert mps.L == 8


def test_MPS_initialize_compress():  # noqa: N802
    key = jax.random.PRNGKey(2)
    mps = MPS_initialize(
        L=8,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=2,
        phys_dim=2,
        compress=True,
    )
    assert mps.L == 8


def test_MPS_initialize_canonical_center():  # noqa: N802
    key = jax.random.PRNGKey(3)
    mps = MPS_initialize(
        L=8,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=4,
        phys_dim=2,
        canonical_center=4,
    )
    assert mps.L == 8


def test_MPS_normalize_with_insert():  # noqa: N802
    key = jax.random.PRNGKey(4)
    mps = MPS_initialize(
        L=6,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=4,
        phys_dim=2,
    )
    mps.normalize(insert=0)  # exercises the single-tensor normalization branch
    assert mps.L == 6


# --- Error paths ---


def test_MPS_initialize_cyclic_noteven_raises():  # noqa: N802
    key = jax.random.PRNGKey(5)
    with pytest.raises(NotImplementedError):
        MPS_initialize(
            L=6,
            initializer=randn(1e-1),
            key=key,
            shape_method="noteven",
            cyclic=True,
        )


def test_MPS_initialize_class_index_too_large_raises():  # noqa: N802
    key = jax.random.PRNGKey(6)
    with pytest.raises(ValueError, match="class_index"):
        MPS_initialize(
            L=4,
            initializer=randn(1e-1),
            key=key,
            shape_method="even",
            class_index=10,
            class_dim=2,
        )


def test_MPS_initialize_compress_noteven_raises():  # noqa: N802
    key = jax.random.PRNGKey(7)
    with pytest.raises(ValueError, match="Compress"):
        MPS_initialize(
            L=6,
            initializer=randn(1e-1),
            key=key,
            shape_method="noteven",
            bond_dim=4,
            phys_dim=2,
            compress=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"add_identity": True}, "add_identity"),
        ({"compress": True}, "compress"),
        ({"insert": 2}, "insert"),
    ],
)
def test_MPS_initialize_rand_unitary_unsupported_options(kwargs, match):  # noqa: N802
    key = jax.random.PRNGKey(8)
    with pytest.raises(ValueError, match=match):
        MPS_initialize(
            L=6,
            initializer=rand_unitary(),
            key=key,
            shape_method="even",
            bond_dim=4,
            phys_dim=2,
            **kwargs,
        )
