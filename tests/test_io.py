"""
Tests for bmodel.io library.

Francesc Font-Clos
Oct 2018
"""
import pytest
from distutils import dir_util
import os
from bmodel.io import topo2interaction
from bmodel.io import edgelist2interaction
import numpy as np


@pytest.fixture
def datadir(tmpdir, request):
    """
    Create a tmpdir with data.

    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.

    taken from:
    https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))
    return tmpdir


# topo2interaction
def test_io_topo2interaction_J(datadir):
    """Test if we can convert a topo file to an interaction matrix."""
    # path to topo file in tmpdir
    path = datadir.join('neg_feedback.topo')
    J, _ = topo2interaction(path)
    expected_J = np.array([[1, -1],
                           [-1, 1]])
    assert np.all(J == expected_J)


def test_io_topo2interaction_nodelabels(datadir):
    """Test if we can convert a topo file to an interaction matrix."""
    # path to topo file in tmpdir
    path = datadir.join('neg_feedback.topo')
    _, node_labels = topo2interaction(path)
    expected_node_labels = np.array(["A", "B"])
    assert np.all(node_labels == expected_node_labels)


def test_io_topo2interaction_noheader(datadir):
    """Test that passing a topo file with no header raises exception."""
    # path to topo file in tmpdir
    path = datadir.join('neg_feedback_noheader.topo')
    # if topo2interaction raises an AssertionError
    # then the test should pass
    try:
        topo2interaction(path)
    except AssertionError:
        return True
    # otherwise it should fail
    assert False


# edgelist2interaction
def test_io_edgelist2interaction_J(datadir):
    """Test if we can convert a csv file to an interaction matrix."""
    # path to csv file in tmpdir
    path = datadir.join('neg_feedback.csv')
    J, _ = edgelist2interaction(path)
    expected_J = np.array([[1, -1],
                           [-1, 1]])
    assert np.all(J == expected_J)


def test_io_edgelist2interaction_nodelabels(datadir):
    """Test if we can convert a csv file to an interaction matrix."""
    # path to csv file in tmpdir
    path = datadir.join('neg_feedback.csv')
    _, node_labels = edgelist2interaction(path)
    expected_node_labels = np.array(["A", "B"])
    assert np.all(node_labels == expected_node_labels)


def test_io_edgelist2interaction_noheader(datadir):
    """Test that passing a topo file with no header raises exception."""
    # path to csv file in tmpdir
    path = datadir.join('neg_feedback_noheader.csv')
    # if edgelist2interaction raises an AssertionError
    # then the test should pass
    try:
        edgelist2interaction(path)
    except AssertionError:
        return True
    # otherwise it should fail
    assert False
