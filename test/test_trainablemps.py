""" Test TrainableMPS initialization. """

import pytest
import jax
import quimb.tensor as qtn
from tn4ml.models.mps import MPS_initialize, trainable_wrapper
from tn4ml.initializers import *
from jax.nn.initializers import *

jax.config.update("jax_enable_x64", True)

@pytest.mark.parametrize("L, initializer, shape_method, bond_dim, phys_dim, cyclic",[
                        (10, orthogonal(), 'even', 5, 2, False),
                        (10, he_normal(), 'even', 5, 2, False),
                        (10, normal(1e-2), 'even', 5, 2, False),
                        (10, uniform(), 'even', 5, 2, False),
                        (10, gramschmidt_init('normal', 1e-3), 'even', 5, 2, False),
                        (10, gramschmidt_init('uniform', 1e-3), 'even', 5, 2, False),
                        (10, identity_init('copy', 1e-7), 'even', 5, 2, False),
                        (10, identity_init('bond', 1e-7), 'even', 5, 10, False),
                        (10, zeros_noise_init(1e-7), 'even', 5, 10, False),
                        (10, gramschmidt_init('normal', 1e-3), 'noteven', 10, 20, False),
                        (10, gramschmidt_init('uniform', 1e-3), 'noteven', 10, 20, False),
                        (10, identity_init('copy', 1e-7), 'noteven', 10, 2, False),
                        (10, identity_init('bond', 1e-7), 'noteven', 10, 2, False),
                        (10, zeros_noise_init(1e-7), 'noteven', 10, 2, False),
                        (10, gramschmidt_init('normal', 1e-3), 'even', 5, 2, True),
                        (10, gramschmidt_init('uniform', 1e-3), 'even', 5, 2, True),
                        (10, identity_init('copy', 1e-7), 'even', 5, 2, True),
                        (10, identity_init('bond', 1e-7), 'even', 5, 2, True),
                        (10, zeros_noise_init(1e-7), 'even', 5, 2, True),
                        ])
def test_MPS_initialize(L, initializer, shape_method, bond_dim, phys_dim, cyclic):
    key = jax.random.PRNGKey(42)
    mps = MPS_initialize(L, initializer, key, shape_method=shape_method, bond_dim=bond_dim, phys_dim=phys_dim, cyclic=cyclic)
    assert mps.norm() == pytest.approx(1.0)

@pytest.mark.parametrize("mps", [qtn.MPS_rand_state(20, bond_dim=2, phys_dim=2)])
def test_trainable_wrapper(mps):
    mps = trainable_wrapper(mps)
    assert mps.norm() == pytest.approx(1.0)