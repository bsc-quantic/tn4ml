"""Test TrainableMPO initialization."""

import jax
import pytest
import quimb.tensor as qtn
from jax.nn.initializers import he_normal, normal, orthogonal, uniform

from tn4ml.initializers import gramschmidt, rand_unitary, randn
from tn4ml.models.mpo import MPO_initialize, trainable_wrapper

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("L", "initializer", "shape_method", "bond_dim", "phys_dim", "cyclic"),
    [
        (10, orthogonal(), "even", 5, (2, 2), False),
        (10, he_normal(), "even", 5, (2, 2), False),
        (10, normal(1e-2), "even", 5, (2, 2), False),
        (10, uniform(), "even", 5, (2, 2), False),
        (10, gramschmidt("normal", 1e-3), "even", 5, (4, 5), False),
        (10, gramschmidt("uniform", 1e-3), "even", 5, (4, 5), False),
        (10, gramschmidt("normal", 1e-3), "noteven", 10, (6, 6), False),
        (10, gramschmidt("uniform", 1e-3), "noteven", 10, (6, 6), False),
        (10, gramschmidt("normal", 1e-3), "even", 5, (10, 5), True),
        (10, gramschmidt("uniform", 1e-3), "even", 5, (5, 5), True),
        (10, randn(1e-7), "even", 5, (5, 5), True),
        (10, randn(0.5), "even", 5, (5, 5), True),
        (10, rand_unitary(), "even", 5, (5, 5), True),
    ],
)
def test_rand_distribution(L, initializer, shape_method, bond_dim, phys_dim, cyclic):
    """Test rand distribution."""
    key = jax.random.PRNGKey(42)
    mpo = MPO_initialize(
        L,
        initializer,
        key,
        shape_method=shape_method,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        cyclic=cyclic,
    )
    assert mpo.norm() == pytest.approx(1.0)


@pytest.mark.parametrize("mpo", [qtn.MPO_rand(20, bond_dim=2, phys_dim=2)])
def test_trainable_wrapper(mpo):
    """Test trainable wrapper."""
    mpo = trainable_wrapper(mpo)
    assert mpo.norm() == pytest.approx(1.0)


# --- Optional features: add_identity, insert, compress, canonical_center ---


def test_MPO_initialize_add_identity():  # noqa: N802
    """Test MPO initialize add identity."""
    key = jax.random.PRNGKey(0)
    mpo = MPO_initialize(
        L=8,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=4,
        phys_dim=(2, 2),
        add_identity=True,
    )
    assert mpo.L == 8


def test_MPO_initialize_insert():  # noqa: N802
    """Test MPO initialize insert."""
    key = jax.random.PRNGKey(1)
    mpo = MPO_initialize(
        L=8,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=4,
        phys_dim=(2, 2),
        insert=2,
    )
    assert mpo.L == 8


def test_MPO_initialize_compress():  # noqa: N802
    """Test MPO initialize compress."""
    key = jax.random.PRNGKey(2)
    mpo = MPO_initialize(
        L=8,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=2,
        phys_dim=(2, 2),
        compress=True,
    )
    assert mpo.L == 8


def test_MPO_initialize_canonical_center():  # noqa: N802
    """Test MPO initialize canonical center."""
    key = jax.random.PRNGKey(3)
    mpo = MPO_initialize(
        L=8,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=4,
        phys_dim=(2, 2),
        canonical_center=4,
    )
    assert mpo.L == 8


def test_MPO_normalize_with_insert():  # noqa: N802
    """Test MPO normalize with insert."""
    key = jax.random.PRNGKey(4)
    mpo = MPO_initialize(
        L=6,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        bond_dim=4,
        phys_dim=(2, 2),
    )
    mpo.normalize(insert=0)  # single-tensor normalization branch
    assert mpo.L == 6


def test_MPO_initialize_cyclic_noteven_raises():  # noqa: N802
    """Test MPO initialize cyclic noteven raises."""
    key = jax.random.PRNGKey(5)
    with pytest.raises(NotImplementedError):
        MPO_initialize(
            L=6,
            initializer=randn(1e-1),
            key=key,
            shape_method="noteven",
            cyclic=True,
        )
