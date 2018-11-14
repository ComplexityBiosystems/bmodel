"""
Tests for the base class.

Francesc Font-Clos
Nov 2018
"""
import pytest
from bmodel.base import Bmodel
import numpy as np
import pandas as pd


@pytest.fixture
def bmodel_neg_feedback():
    """Create simple network for testing."""
    J = np.array([[0, -1], [-1, 0]])
    node_labels = ["node0", "node1"]
    bmodel = Bmodel(J=J, node_labels=node_labels, maxT=100)
    return bmodel


def test_base_bmodel_init():
    """Test that bmodel objects can be instantiated."""
    J = np.array([[0, -1], [-1, 0]])
    bmodel = Bmodel(J=J)
    assert isinstance(bmodel, Bmodel)


def test_base_bmodel_node_labels(bmodel_neg_feedback):
    """Test that node labels are correctly passed onto the model."""
    bmodel = bmodel_neg_feedback
    for a, b in zip(list(bmodel.ss.columns), bmodel.node_labels):
        assert a == b


def test_base_bmodel_runs(bmodel_neg_feedback):
    """Test that the model can run simulations."""
    bmodel = bmodel_neg_feedback
    bmodel.runs(20)
    assert len(bmodel.ss) > 0


def test_base_bmodel_initial_conditions(bmodel_neg_feedback):
    """Test that the model is using random initial conditions."""
    bmodel = bmodel_neg_feedback
    bmodel.runs(1000)
    df_ic = pd.DataFrame(
        bmodel._initial_conditions,
        columns=bmodel.node_labels
    )
    assert df_ic.mean().abs().max() < 0.05


def test_base_bmodel_perturbe_ss(bmodel_neg_feedback):
    """Test that the perturbations ss df has right shape."""
    bmodel = bmodel_neg_feedback
    bmodel.runs(20)
    n_runs = 3
    config = bmodel.ss.values[0]
    for node_to_switch in bmodel.node_labels:
        for switch_to in ["ON", "OFF"]:
            bmodel.perturbe(
                initial_condition=config,
                node_to_switch=node_to_switch,
                switch_to=switch_to,
                n_runs=n_runs
            )
    assert bmodel._perturbations_ss.shape == (n_runs * bmodel.N * 2, bmodel.N)


def test_base_bmodel_perturbe_ic(bmodel_neg_feedback):
    """Test that the perturbations ic df has right shape."""
    bmodel = bmodel_neg_feedback
    bmodel.runs(20)
    n_runs = 3
    config = bmodel.ss.values[0]
    for node_to_switch in bmodel.node_labels:
        for switch_to in ["ON", "OFF"]:
            bmodel.perturbe(
                initial_condition=config,
                node_to_switch=node_to_switch,
                switch_to=switch_to,
                n_runs=n_runs
            )
    assert bmodel._perturbations_ic.shape == (n_runs * bmodel.N * 2, bmodel.N)


def test_base_bmodel_perturbe_meta(bmodel_neg_feedback):
    """Test that the perturbations meta df has right shape."""
    bmodel = bmodel_neg_feedback
    bmodel.runs(20)
    n_runs = 3
    config = bmodel.ss.values[0]
    for node_to_switch in bmodel.node_labels:
        for switch_to in ["ON", "OFF"]:
            bmodel.perturbe(
                initial_condition=config,
                node_to_switch=node_to_switch,
                switch_to=switch_to,
                n_runs=n_runs
            )
    assert bmodel._perturbations_meta.shape == (n_runs * bmodel.N * 2, 2)
