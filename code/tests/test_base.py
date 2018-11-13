"""
Tests for the base class.

Francesc Font-Clos
Nov 2018
"""
import pytest
from bmodel.base import Bmodel
import numpy as np


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
