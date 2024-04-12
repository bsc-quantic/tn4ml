""" Test SMPO initialization """

import pytest
import jax
from tn4ml.models.smpo import SpacedMatrixProductOperator, SMPO_initialize
from tn4ml.initializers import *
from jax.nn.initializers import *

jax.config.update("jax_enable_x64", True)
@pytest.mark.parametrize("L, initializer, shape_method, spacing, bond_dim, phys_dim, cyclic", [
        (10, gramschmidt_init('normal', 1e-3), 'even', 5, 5, (2,2), False),
        (10, gramschmidt_init('uniform', 1e-3), 'even', 5, 5, (2,2), False),
        (10, identity_init('copy', 1e-7), 'even', 5, 5, (2,2), False),
        (10, identity_init('bond', 1e-7), 'even', 5, 5, (2,2), False),
        (10, zeros_noise_init(1e-7), 'even', 5, 5, (2,2), False),
        (10, gramschmidt_init('normal', 1e-3), 'noteven', 10, 10, (2,2), False),
        (10, gramschmidt_init('uniform', 1e-3), 'noteven', 10, 10, (2,2), False),
        (10, identity_init('copy', 1e-7), 'noteven', 10, 10, (2,2), False),
        (10, identity_init('bond', 1e-7), 'noteven', 10, 10, (2,2), False),
        (10, zeros_noise_init(1e-7), 'noteven', 10, 10, (2,2), False),
        (10, gramschmidt_init('normal', 1e-3), 'even', 5, 5, (2,2), True),
        (10, gramschmidt_init('uniform', 1e-3), 'even', 5, 5, (2,2), True),
        (10, identity_init('copy', 1e-7), 'even', 5, 5, (2,2), True),
        (10, identity_init('bond', 1e-7), 'even', 5, 5, (2,2), True),
        (10, zeros_noise_init(1e-7), 'even', 5, 5, (2,2), True),
        (10, gramschmidt_init('normal', 1e-3), 'even', 10, 10, (2,2), True),
        (10, gramschmidt_init('uniform', 1e-3), 'even', 10, 10, (2,2), True),
        (10, identity_init('copy', 1e-7), 'even', 10, 10, (2,2), True),
        (10, identity_init('bond', 1e-7), 'even', 10, 10, (2,2), True),
        (10, orthogonal(), 'even', 5, 5, (2,2), False),
        (10, he_normal(), 'even', 5, 5, (2,2), False),
        (10, normal(1e-2), 'even', 5, 5, (2,2), False),
        (10, uniform(), 'even', 5, 5, (2,2), False),

])
def test_SMPO_initialize(L, initializer, shape_method, spacing, bond_dim, phys_dim, cyclic):
    key = jax.random.PRNGKey(42)
    print(phys_dim)
    print(cyclic)
    smpo = SMPO_initialize(L=L, initializer=initializer, key=key, shape_method=shape_method, spacing=spacing, bond_dim=bond_dim, phys_dim=phys_dim, cyclic=cyclic)

    assert smpo.norm() == pytest.approx(1.0)