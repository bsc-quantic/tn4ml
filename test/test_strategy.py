"""Test strategy classes."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tn4ml.strategy import Strategy, Sweeps, Global, _check_model, _get_inds_for_split
from tn4ml.models.mps import MPS_initialize
from tn4ml.models.smpo import SMPO_initialize
from tn4ml.initializers import randn

jax.config.update("jax_enable_x64", True)


# --- Strategy base class ---

def test_strategy_default():
    s = Strategy()
    assert s.renormalize is False

def test_strategy_renormalize():
    s = Strategy(renormalize=True)
    assert s.renormalize is True


# --- Sweeps ---

def test_sweeps_default():
    s = Sweeps()
    assert s.grouping == 2
    assert s.two_way is True

def test_sweeps_invalid_grouping_gt2():
    with pytest.raises(ValueError, match="grouping"):
        Sweeps(grouping=3)

def test_sweeps_invalid_grouping_eq1():
    with pytest.raises(ValueError, match="grouping == 1"):
        Sweeps(grouping=1)

def test_sweeps_one_way():
    s = Sweeps(two_way=False)
    assert s.two_way is False

def test_sweeps_iterate_sites():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(L=5, initializer=randn(1e-2), key=key,
                           shape_method='even', bond_dim=3, phys_dim=2, cyclic=False)
    s = Sweeps()
    sites = list(s.iterate_sites(model))
    # Forward: (0,1), (1,2), (2,3), (3,4)
    # Backward: (4,3), (3,2), (2,1), (1,0)
    assert len(sites) == 8  # 4 forward + 4 backward

def test_sweeps_iterate_sites_one_way():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(L=5, initializer=randn(1e-2), key=key,
                           shape_method='even', bond_dim=3, phys_dim=2, cyclic=False)
    s = Sweeps(two_way=False)
    sites = list(s.iterate_sites(model))
    assert len(sites) == 4  # forward only


# --- _check_model ---

def test_check_model_valid():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(L=5, initializer=randn(1e-2), key=key,
                           shape_method='even', bond_dim=3, phys_dim=2, cyclic=False)
    # Should not raise
    _check_model(model)

def test_check_model_invalid():
    with pytest.raises(TypeError, match="necessary methods"):
        _check_model(object())


# --- _get_inds_for_split ---

def test_get_inds_for_split_basic():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(L=5, initializer=randn(1e-2), key=key,
                           shape_method='even', bond_dim=3, phys_dim=2, cyclic=False)
    left_inds, right_inds, bond = _get_inds_for_split(
        model.ind_map, sitel=1, siter=2, nsites=5
    )
    assert bond == "bond_1"
    assert "k1" in left_inds
    assert "k2" in right_inds


# --- Sweeps variants ---

def test_sweeps_one_way_grouping2_attrs():
    s = Sweeps(two_way=False, grouping=2)
    assert s.two_way is False
    assert s.grouping == 2

def test_sweeps_default_has_inds_order():
    s = Sweeps()
    assert hasattr(s, 'inds_order')
    assert isinstance(s.inds_order, dict)
    assert len(s.inds_order) == 0

def test_sweeps_one_way_has_inds_order():
    s = Sweeps(two_way=False, grouping=2)
    assert hasattr(s, 'inds_order')
    assert isinstance(s.inds_order, dict)


# --- iterate_sites content ---

def test_sweeps_iterate_sites_two_way_content():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(L=5, initializer=randn(1e-2), key=key,
                           shape_method='even', bond_dim=3, phys_dim=2, cyclic=False)
    s = Sweeps(two_way=True, grouping=2)
    sites = list(s.iterate_sites(model))
    # Forward: left-to-right pairs
    assert sites[0] == (0, 1)
    assert sites[1] == (1, 2)
    assert sites[2] == (2, 3)
    assert sites[3] == (3, 4)
    # Backward: right-to-left pairs
    assert sites[4] == (4, 3)
    assert sites[5] == (3, 2)
    assert sites[6] == (2, 1)
    assert sites[7] == (1, 0)

def test_sweeps_iterate_sites_one_way_content():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(L=5, initializer=randn(1e-2), key=key,
                           shape_method='even', bond_dim=3, phys_dim=2, cyclic=False)
    s = Sweeps(two_way=False, grouping=2)
    sites = list(s.iterate_sites(model))
    assert sites == [(0, 1), (1, 2), (2, 3), (3, 4)]


# --- prehook / posthook ---

def test_sweeps_prehook_populates_inds_order():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(L=5, initializer=randn(1e-2), key=key,
                           shape_method='even', bond_dim=3, phys_dim=2, cyclic=False)
    s = Sweeps()
    sites = (0, 1)
    assert sites not in s.inds_order
    s.prehook(model, sites)
    assert sites in s.inds_order
    assert len(s.inds_order[sites]) > 0

def test_sweeps_prehook_contracts_tensors():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(L=5, initializer=randn(1e-2), key=key,
                           shape_method='even', bond_dim=3, phys_dim=2, cyclic=False)
    s = Sweeps()
    n_before = len(model.tensors)
    s.prehook(model, (0, 1))
    assert len(model.tensors) == n_before - 1

def test_sweeps_prehook_posthook_roundtrip_tensor_count():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(L=5, initializer=randn(1e-2), key=key,
                           shape_method='even', bond_dim=3, phys_dim=2, cyclic=False)
    s = Sweeps()
    n_before = len(model.tensors)
    sites = (0, 1)
    s.prehook(model, sites)
    s.posthook(model, sites)
    assert len(model.tensors) == n_before

def test_sweeps_posthook_restores_site_tags():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(L=5, initializer=randn(1e-2), key=key,
                           shape_method='even', bond_dim=3, phys_dim=2, cyclic=False)
    s = Sweeps()
    sites = (0, 1)
    s.prehook(model, sites)
    s.posthook(model, sites)
    # Both site tensors must be retrievable by tag after the split
    assert len(model.select_tensors(model.site_tag(0))) == 1
    assert len(model.select_tensors(model.site_tag(1))) == 1

def test_sweeps_prehook_posthook_backward_sites():
    key = jax.random.PRNGKey(42)
    model = MPS_initialize(L=5, initializer=randn(1e-2), key=key,
                           shape_method='even', bond_dim=3, phys_dim=2, cyclic=False)
    s = Sweeps(two_way=True)
    n_before = len(model.tensors)
    # Simulate a backward step: sites come in reversed order
    sites = (4, 3)
    s.prehook(model, sites)
    assert len(model.tensors) == n_before - 1
    s.posthook(model, sites)
    assert len(model.tensors) == n_before
    assert len(model.select_tensors(model.site_tag(3))) == 1
    assert len(model.select_tensors(model.site_tag(4))) == 1
