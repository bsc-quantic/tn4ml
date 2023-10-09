""" Test SMPO initialization. """

import pytest
import jax
from tn4ml.models import SpacedMatrixProductOperator
jax.config.update("jax_enable_x64", True)

@pytest.mark.parametrize("n", [20, 400])
@pytest.mark.parametrize("spacing", [5, 20])
@pytest.mark.parametrize("bond_dim", [5, 100])
@pytest.mark.parametrize("phys_dim", [(2,2)])
@pytest.mark.parametrize("seed", [None, 123])
def test_rand_orthogonal(n, spacing, bond_dim, phys_dim, seed):
    smpo = SpacedMatrixProductOperator.rand_orthogonal(n, spacing=spacing, bond_dim=bond_dim, seed=seed, phys_dim=phys_dim)
    assert smpo.norm() == pytest.approx(1.0)

@pytest.mark.parametrize("n", [20, 400])
@pytest.mark.parametrize("spacing", [5, 20])
@pytest.mark.parametrize("bond_dim", [5, 100])
@pytest.mark.parametrize("phys_dim", [(2,2)])
@pytest.mark.parametrize("seed", [None, 123])
def test_rand_distribution(n, spacing, bond_dim, phys_dim, seed):
    smpo = SpacedMatrixProductOperator.rand_distribution(n, spacing=spacing, bond_dim=bond_dim, seed=seed, phys_dim=phys_dim)
    assert smpo.norm() == pytest.approx(1.0)