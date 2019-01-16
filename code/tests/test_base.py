"""
Tests for the base class.

Francesc Font-Clos
Nov 2018
"""

from bmodel.base import Bmodel
from bmodel.generate import random_interaction_matrix
from bmodel.exceptions import IndicatorError

import numpy as np
import pandas as pd
import pytest


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


def test_base_bmodel_node_labels(bmodel_neg_feedback: Bmodel):
    """Test that node labels are correctly passed onto the model."""
    bmodel = bmodel_neg_feedback
    for a, b in zip(list(bmodel.steady_states.columns), bmodel.node_labels):
        assert a == b


def test_base_bmodel_runs(bmodel_neg_feedback: Bmodel):
    """Test that the model can run simulations."""
    bmodel = bmodel_neg_feedback
    bmodel.runs(20)
    assert len(bmodel.steady_states) > 0


def test_base_bmodel_runs_fast(bmodel_neg_feedback: Bmodel):
    """Test that the model can run simulations in fast mode."""
    bmodel = bmodel_neg_feedback
    bmodel.runs(20, fast=True)
    assert len(bmodel.steady_states) > 0


def test_base_bmodel_initial_conditions(bmodel_neg_feedback: Bmodel):
    """Test that the model is using random initial conditions."""
    bmodel = bmodel_neg_feedback
    bmodel.runs(1000)
    df_ic = pd.DataFrame(
        bmodel._initial_conditions,
        columns=bmodel.node_labels
    )
    assert df_ic.mean().abs().max() < 0.1


def test_base_bmodel_perturbe_ss(bmodel_neg_feedback: Bmodel):
    """Test that the perturbations ss df has right shape."""
    bmodel = bmodel_neg_feedback
    bmodel.runs(20)
    n_runs = 3
    config = bmodel.steady_states.values[0]
    for node_to_switch in bmodel.node_labels:
        for switch_to in ["ON", "OFF"]:
            bmodel.perturbe(
                initial_condition=config,
                node_to_switch=node_to_switch,
                switch_to=switch_to,
                n_runs=n_runs
            )
    assert bmodel._perturbations_ss.shape == (n_runs * bmodel.N * 2, bmodel.N)


def test_base_bmodel_perturbe_ic(bmodel_neg_feedback: Bmodel):
    """Test that the perturbations ic df has right shape."""
    bmodel = bmodel_neg_feedback
    bmodel.runs(20)
    n_runs = 3
    config = bmodel.steady_states.values[0]
    for hold in [True, False]:
        for switch_to in ["ON", "OFF"]:
            for node_to_switch in bmodel.node_labels:
                bmodel.perturbe(
                    initial_condition=config,
                    node_to_switch=node_to_switch,
                    switch_to=switch_to,
                    n_runs=n_runs,
                    hold=hold
                )
    assert bmodel._perturbations_ic.shape == (n_runs * bmodel.N * 4, bmodel.N)


def test_base_bmodel_perturbe_meta(bmodel_neg_feedback: Bmodel):
    """Test that the perturbations meta df has right shape."""
    bmodel = bmodel_neg_feedback
    bmodel.runs(20)
    n_runs = 3
    config = bmodel.steady_states.values[0]
    for hold in [True, False]:
        for switch_to in ["ON", "OFF"]:
            for node_to_switch in bmodel.node_labels:
                bmodel.perturbe(
                    initial_condition=config,
                    node_to_switch=node_to_switch,
                    switch_to=switch_to,
                    n_runs=n_runs,
                    hold=hold
                )
    assert bmodel._perturbations_meta.shape == (n_runs * bmodel.N * 4, 3)


def test_base_bmodel_get_perturbations():
    """Test that we retrieve the right number of perturbations"""
    n_nodes = 16
    n_runs_pert = 5
    bmodel = Bmodel(J=random_interaction_matrix(N=n_nodes))
    bmodel.runs(20)
    for config in bmodel.steady_states.values:
        for hold in [True, False]:
            for node_to_switch in bmodel.node_labels:
                for switch_to in ["ON", "OFF"]:
                    bmodel.perturbe(
                        initial_condition=config,
                        node_to_switch=node_to_switch,
                        switch_to=switch_to,
                        n_runs=n_runs_pert,
                        hold=hold
                    )

    for hold in [True, False]:
        for switched_node in bmodel.node_labels:
            for switch_to in ["ON", "OFF"]:
                perturbations = bmodel.get_perturbations(
                    switched_node=switched_node,
                    switch_to=switch_to,
                    hold=hold
                )
                n_ics = (bmodel.steady_states[switched_node] ==
                         {"ON": -1, "OFF": 1}[switch_to]).sum()
                assert perturbations.shape == (n_ics * n_runs_pert, n_nodes)


def test_base_bmodel_hold():
    """Test the hold option"""
    n_nodes = 12
    n_runs_pert = 4
    bmodel = Bmodel(J=random_interaction_matrix(N=n_nodes))
    bmodel.runs(16)
    hold = True
    for config in bmodel.steady_states.values:
        for node_to_switch in bmodel.node_labels:
            for switch_to in ["ON", "OFF"]:
                bmodel.perturbe(
                    initial_condition=config,
                    node_to_switch=node_to_switch,
                    switch_to=switch_to,
                    n_runs=n_runs_pert,
                    hold=hold
                )

    # If node is holded, its final state must be what we asked for
    for switched_node in bmodel.node_labels:
        for switch_to in ["ON", "OFF"]:
            perturbations = bmodel.get_perturbations(
                switched_node=switched_node,
                switch_to=switch_to,
                hold=hold
            )
            assert np.all(perturbations[switched_node] ==
                          {"ON": 1, "OFF": -1}[switch_to])


def test_base_bmodel_no_hold():
    """Test the no hold option"""
    n_nodes = 12
    n_runs_pert = 4
    bmodel = Bmodel(J=random_interaction_matrix(N=n_nodes))
    bmodel.runs(16)
    hold = False
    for config in bmodel.steady_states.values:
        for node_to_switch in bmodel.node_labels:
            for switch_to in ["ON", "OFF"]:
                bmodel.perturbe(
                    initial_condition=config,
                    node_to_switch=node_to_switch,
                    switch_to=switch_to,
                    n_runs=n_runs_pert,
                    hold=hold
                )

    # if node is not holded, then final state can or cannot be what we asked for
    # this is difficult to test
    checks = []
    for switched_node in bmodel.node_labels:
        for switch_to in ["ON", "OFF"]:
            perturbations = bmodel.get_perturbations(
                switched_node=switched_node,
                switch_to=switch_to,
                hold=hold
            )
            checks.append((perturbations[switched_node] ==
                           {"ON": -1, "OFF": 1}[switch_to]).values)
    checks = list(np.concatenate(checks))
    checks = sorted(list(set(checks)))
    assert checks == [False, True]


def test_base_bmodel_indicator_is_set():
    n_nodes = 16
    node_labels = [
        "node_%d" % i
        for i in range(n_nodes)
    ]
    # let us get at least two indicator nodes
    num_indicators = 0
    while num_indicators <= 1:
        J = random_interaction_matrix(N=n_nodes)
        indicator_nodes = [
            node
            for i, node in enumerate(node_labels)
            if np.all(J[:, i] == 0)
        ]
        num_indicators = len(indicator_nodes)
    bmodel = Bmodel(
        J=J,
        node_labels=node_labels,
        indicator_nodes=indicator_nodes
    )
    assert bmodel.indicator_nodes == list(indicator_nodes)


def test_base_bmodel_indicator_can_fail():
    n_nodes = 16
    J = random_interaction_matrix(N=n_nodes)
    node_labels = [
        "node_%d" % i
        for i in range(n_nodes)
    ]
    indicator_nodes = ["node_a", "node_b"]
    with pytest.raises(IndicatorError):
        _ = Bmodel(
            J=J,
            node_labels=node_labels,
            indicator_nodes=indicator_nodes
        )


def test_base_bmodel_indicator_unique():
    """Test that we cannot pass non-unique indicators"""
    n_nodes = 16
    J = random_interaction_matrix(N=n_nodes)
    node_labels = [
        "node_%d" % i
        for i in range(n_nodes)
    ]
    indicator_nodes = ["node_0", "node_1", "node_0"]
    with pytest.raises(IndicatorError):
        _ = Bmodel(
            J=J,
            node_labels=node_labels,
            indicator_nodes=indicator_nodes
        )


def test_base_bmodel_indicator_idx():
    """Test that we correctly save the positions of indicator nodes"""
    n_nodes = 4
    J = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    node_labels = [
        "node_%d" % i
        for i in range(n_nodes)
    ]
    indicator_nodes = [
        "node_0",
        "node_3"
    ]
    bmodel = Bmodel(
        J=J,
        node_labels=node_labels,
        indicator_nodes=indicator_nodes
    )
    assert bmodel._indicator_idx == [0, 3]


def test_base_bmodel_indicator_not_inputs():
    """Test that we cannot pass as indicator nodes that are input of other nodes"""
    J = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    # node 1 cannot be indicator
    with pytest.raises(IndicatorError):
        _ = Bmodel(
            J=J,
            indicator_nodes=[1]
        )
    # node 2 cannot be indicator
    with pytest.raises(IndicatorError):
        _ = Bmodel(
            J=J,
            indicator_nodes=[2]
        )
    # instead node 0 can
    _ = Bmodel(
        J=J,
        indicator_nodes=[0]
    )
    assert True


def test_base_bmodel_indicator_not_inputs_using_labels():
    """Test that we cannot pass as indicator nodes that are input of other nodes, using string labels"""
    J = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    # node 1 cannot be indicator
    with pytest.raises(IndicatorError):
        _ = Bmodel(
            J=J,
            node_labels=["a", "b", "c"],
            indicator_nodes=["b"]
        )
    # node 2 cannot be indicator
    with pytest.raises(IndicatorError):
        _ = Bmodel(
            J=J,
            node_labels=["a", "b", "c"],
            indicator_nodes=["c"]
        )
    # instead node 0 can
    _ = Bmodel(
        J=J,
        node_labels=["a", "b", "c"],
        indicator_nodes=["a"]
    )
    assert True
