""" Test TrainableMPS initialization. """

import pytest
import jax
from tn4ml.models import TrainableMatrixProductState
jax.config.update("jax_enable_x64", True)

@pytest.mark.parametrize("L", [20, 400])
@pytest.mark.parametrize("bond_dim", [5, 50])
@pytest.mark.parametrize("phys_dim", [2, 10])
def test_rand_distribution(L, bond_dim, phys_dim):
    mps = TrainableMatrixProductState.rand_state(L, bond_dim=bond_dim, phys_dim=phys_dim)
    assert mps.norm() == pytest.approx(1.0)


@pytest.mark.parametrize("L", [20, 400])
@pytest.mark.parametrize("bond_dim", [5, 50])
@pytest.mark.parametrize("phys_dim", [2])
@pytest.mark.parametrize("seed", [None, 123])
def test_rand_orthogonal(L, bond_dim, phys_dim, seed):
    mps = TrainableMatrixProductState.rand_orthogonal(L, bond_dim=bond_dim, phys_dim=phys_dim, seed=seed)
    assert mps.norm() == pytest.approx(1.0)
