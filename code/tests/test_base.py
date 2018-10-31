import pytest
from bmodel.base import Bmodel
import numpy as np


@pytest.fixture
def bmodel_neg_feedback():
    J = np.array([[0, -1], [-1, 0]])
    node_labels = ["node0", "node1"]
    bmodel = Bmodel(J=J, node_labels=node_labels)
    bmodel.runs(10)
    return bmodel


def test_base_bmodel_init():
    """Test that bmodel objects can be instantiated"""
    J = np.array([[0, -1], [-1, 0]])
    bmodel = Bmodel(J=J)
    assert isinstance(bmodel, Bmodel)


def test_base_bmodel_node_labels(bmodel_neg_feedback):
    bmodel = bmodel_neg_feedback
    for a, b in zip(list(bmodel.ss.columns), bmodel.node_labels):
        assert a == b
