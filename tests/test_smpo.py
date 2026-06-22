"""Test SMPO initialization"""

import jax
import pytest
from jax.nn.initializers import he_normal, normal, orthogonal, uniform

from tn4ml.initializers import gramschmidt, randn
from tn4ml.models.smpo import SMPO_initialize

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("L", "initializer", "shape_method", "spacing", "bond_dim", "phys_dim", "cyclic"),
    [
        (10, gramschmidt("normal", 1e-3), "even", 5, 5, (2, 2), False),
        (10, gramschmidt("uniform", 1e-3), "even", 5, 5, (2, 2), False),
        (10, randn(1e-7), "even", 5, 5, (2, 2), False),
        (10, gramschmidt("normal", 1e-3), "noteven", 10, 10, (2, 2), False),
        (10, gramschmidt("uniform", 1e-3), "noteven", 10, 10, (2, 2), False),
        (10, gramschmidt("normal", 1e-3), "even", 5, 5, (2, 2), True),
        (10, gramschmidt("uniform", 1e-3), "even", 5, 5, (2, 2), True),
        (10, randn(1e-7), "even", 5, 5, (2, 2), True),
        (10, gramschmidt("normal", 1e-3), "even", 10, 10, (2, 2), True),
        (10, gramschmidt("uniform", 1e-3), "even", 10, 10, (2, 2), True),
        (10, randn(1e-7), "even", 10, 10, (2, 2), True),
        (10, orthogonal(), "even", 5, 5, (2, 2), False),
        (10, he_normal(), "even", 5, 5, (2, 2), False),
        (10, normal(1e-2), "even", 5, 5, (2, 2), False),
        (10, uniform(), "even", 5, 5, (2, 2), False),
    ],
)
def test_SMPO_initialize(  # noqa: N802
    L, initializer, shape_method, spacing, bond_dim, phys_dim, cyclic
):
    key = jax.random.PRNGKey(42)
    print(phys_dim)
    print(cyclic)
    smpo = SMPO_initialize(
        L=L,
        initializer=initializer,
        key=key,
        shape_method=shape_method,
        spacing=spacing,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        cyclic=cyclic,
    )

    assert smpo.norm() == pytest.approx(1.0)


# --- SMPO properties ---


def test_SMPO_spacing():  # noqa: N802
    key = jax.random.PRNGKey(42)
    smpo = SMPO_initialize(
        L=10,
        initializer=gramschmidt("normal", 1e-3),
        key=key,
        shape_method="even",
        spacing=5,
        bond_dim=5,
        phys_dim=(2, 2),
        cyclic=False,
    )
    assert smpo.spacing == 5


def test_SMPO_get_orders():  # noqa: N802
    key = jax.random.PRNGKey(42)
    smpo = SMPO_initialize(
        L=10,
        initializer=gramschmidt("normal", 1e-3),
        key=key,
        shape_method="even",
        spacing=5,
        bond_dim=5,
        phys_dim=(2, 2),
        cyclic=False,
    )
    orders = smpo.get_orders()
    assert len(orders) == 10


# --- SMPO norm ---


def test_SMPO_norm():  # noqa: N802
    key = jax.random.PRNGKey(42)
    smpo = SMPO_initialize(
        L=10,
        initializer=gramschmidt("normal", 1e-3),
        key=key,
        shape_method="even",
        spacing=5,
        bond_dim=5,
        phys_dim=(2, 2),
        cyclic=False,
    )
    n = smpo.norm()
    assert n == pytest.approx(1.0)


# --- SMPO normalize ---


def test_SMPO_normalize():  # noqa: N802
    key = jax.random.PRNGKey(42)
    smpo = SMPO_initialize(
        L=10,
        initializer=randn(1.0),
        key=key,
        shape_method="even",
        spacing=5,
        bond_dim=5,
        phys_dim=(2, 2),
        cyclic=False,
    )
    # Scale up tensors
    for t in smpo.tensors:
        t.modify(data=t.data * 3.0)
    smpo.normalize()
    assert smpo.norm() == pytest.approx(1.0, abs=1e-4)


# --- SMPO apply ---


def test_SMPO_apply():  # noqa: N802
    import quimb.tensor as qtn

    key = jax.random.PRNGKey(42)
    smpo = SMPO_initialize(
        L=10,
        initializer=orthogonal(),
        key=key,
        shape_method="even",
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        cyclic=False,
    )
    mps = qtn.MPS_rand_state(10, bond_dim=2, phys_dim=2)
    result = smpo.apply(mps)
    assert result is not None


# --- Optional features: add_identity, insert, compress, canonical_center ---


def test_SMPO_initialize_add_identity():  # noqa: N802
    key = jax.random.PRNGKey(0)
    smpo = SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        add_identity=True,
        add_to_output=True,
    )
    assert smpo.L == 10


def test_SMPO_initialize_insert():  # noqa: N802
    key = jax.random.PRNGKey(1)
    smpo = SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        insert=2,
    )
    assert smpo.L == 10


def test_SMPO_initialize_compress():  # noqa: N802
    key = jax.random.PRNGKey(2)
    smpo = SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        spacing=2,
        bond_dim=2,
        phys_dim=(2, 2),
        compress=True,
    )
    assert smpo.L == 10


def test_SMPO_initialize_canonical_center():  # noqa: N802
    key = jax.random.PRNGKey(3)
    smpo = SMPO_initialize(
        L=10,
        initializer=randn(1e-1),
        key=key,
        shape_method="even",
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        canonical_center=4,
    )
    assert smpo.L == 10


# --- Error paths ---


def test_SMPO_initialize_spacing_one_raises():  # noqa: N802
    key = jax.random.PRNGKey(5)
    with pytest.raises(ValueError, match="Spacing must be"):
        SMPO_initialize(
            L=10,
            initializer=randn(1e-1),
            key=key,
            shape_method="even",
            spacing=1,
            bond_dim=4,
            phys_dim=(2, 2),
        )


def test_SMPO_initialize_output_inds_without_first_raises():  # noqa: N802
    key = jax.random.PRNGKey(6)
    with pytest.raises(ValueError, match="First tensor"):
        SMPO_initialize(
            L=10,
            initializer=randn(1e-1),
            key=key,
            shape_method="even",
            bond_dim=4,
            phys_dim=(2, 2),
            output_inds=[3, 6],
        )


def test_SMPO_initialize_cyclic_noteven_raises():  # noqa: N802
    key = jax.random.PRNGKey(7)
    with pytest.raises(NotImplementedError):
        SMPO_initialize(
            L=10,
            initializer=randn(1e-1),
            key=key,
            shape_method="noteven",
            spacing=2,
            cyclic=True,
        )
