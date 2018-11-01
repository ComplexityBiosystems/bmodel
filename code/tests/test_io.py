"""
Tests for bmodel.io library.

Francesc Font-Clos
Oct 2018
"""
import pytest
from distutils import dir_util
import os
from bmodel.io import topo2interaction
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


def test_io_topo2interaction(datadir):
    """Test if we can convert a topo file to an interaction matrix."""
    # path to topo file in tmpdir
    path = datadir.join('neg_feedback.topo')
    J = topo2interaction(path)
    expected_J = np.array([[1, -1],
                           [-1, 1]])
    assert np.all(J == expected_J)


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
