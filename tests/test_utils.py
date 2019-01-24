"""
Tests for bmodel.utils library.

Francesc Font-Clos
Oct 2018
"""
import numpy as np

from bmodel.utils import check_interaction_matrix


def test_utils_check_interaction_matrix_pass():
    """Test that a valid interaction matrix passes the checker."""
    J = np.array([[0, 1], [0, 0]])
    assert check_interaction_matrix(J)


def test_utils_check_interaction_matrix_fail():
    """Test that an invalid interaction matrix fails the checker."""
    J = np.array([[0, 1], [-1, 2]])
    assert not check_interaction_matrix(J)


def test_utils_check_interaction_matrix_zeros():
    """Test that a zero's interaction matrix fails the checker."""
    J = np.array([[0, 0], [0, 0]])
    assert not check_interaction_matrix(J)


def test_utils_check_interaction_matrix_list():
    """Test that a valid interaction matrix passes the checker."""
    J = ([[0, 1], [0, 0]])
    assert not check_interaction_matrix(J)
