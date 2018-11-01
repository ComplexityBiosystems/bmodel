"""
Tests for the generate library.

Francesc Font-Clos
Nov 2018
"""
from bmodel.generate import random_interaction_matrix
import numpy as np
from scipy.sparse.csgraph import connected_components


def test_generate_random_interaction_matrix_N():
    """Test that the generate matrix has right N."""
    J = random_interaction_matrix(N=100, avk=3, mu=0.8, connected=True)
    assert J.shape == (100, 100)


def test_generate_random_interaction_matrix_avk3_N10():
    """Test that the generate matrix has right avk."""
    N = 10
    avk = 3
    avks = [np.abs(random_interaction_matrix(N=N, avk=avk)).mean() * N
            for _ in range(1000)]
    rel_err = np.abs(np.mean(avks) / avk - 1)
    assert rel_err < 0.05


def test_generate_random_interaction_matrix_avk5_N10():
    """Test that the generate matrix has right avk."""
    N = 10
    avk = 5
    avks = [np.abs(random_interaction_matrix(N=N, avk=avk)).mean() * N
            for _ in range(1000)]
    rel_err = np.abs(np.mean(avks) / avk - 1)
    assert rel_err < 0.05


def test_generate_random_interaction_matrix_avk3_N100():
    """Test that the generate matrix has right avk."""
    N = 100
    avk = 3
    avks = [np.abs(random_interaction_matrix(N=N, avk=avk)).mean() * N
            for _ in range(1000)]
    rel_err = np.abs(np.mean(avks) / avk - 1)
    assert rel_err < 0.05


def test_generate_random_interaction_matrix_avk5_N100():
    """Test that the generate matrix has right avk."""
    N = 100
    avk = 5
    avks = [np.abs(random_interaction_matrix(N=N, avk=avk)).mean() * N
            for _ in range(1000)]
    rel_err = np.abs(np.mean(avks) / avk - 1)
    assert rel_err < 0.05


def test_generate_random_interaction_matrix_mu08():
    """Test that the generate matrix has right mu."""
    N = 40
    avk = 3
    mu = 0.8
    mus = []
    for _ in range(1000):
        J = random_interaction_matrix(N=N, avk=avk, mu=mu)
        mus.append(np.sum(J == 1) / np.sum(J != 0))
    rel_err = np.abs(np.mean(mus) / mu - 1)
    assert rel_err < 0.01


def test_generate_random_interaction_matrix_mu06():
    """Test that the generate matrix has right mu."""
    N = 40
    avk = 3
    mu = 0.6
    mus = []
    for _ in range(1000):
        J = random_interaction_matrix(N=N, avk=avk, mu=mu)
        mus.append(np.sum(J == 1) / np.sum(J != 0))
    rel_err = np.abs(np.mean(mus) / mu - 1)
    assert rel_err < 0.01


def test_generate_random_interaction_matrix_mu04():
    """Test that the generate matrix has right mu."""
    N = 40
    avk = 3
    mu = 0.4
    mus = []
    for _ in range(1000):
        J = random_interaction_matrix(N=N, avk=avk, mu=mu)
        mus.append(np.sum(J == 1) / np.sum(J != 0))
    rel_err = np.abs(np.mean(mus) / mu - 1)
    assert rel_err < 0.01


def test_generate_random_interaction_matrix_connected_N10():
    """Test that the generate matrix is connected."""
    J = random_interaction_matrix(N=10, avk=3, mu=0.8, connected=True)
    n, _ = connected_components(np.abs(J))
    assert n == 1


def test_generate_random_interaction_matrix_connected_N40():
    """Test that the generate matrix is connected."""
    J = random_interaction_matrix(N=40, avk=3, mu=0.8, connected=True)
    n, _ = connected_components(np.abs(J))
    assert n == 1


def test_generate_random_interaction_matrix_connected_N100():
    """Test that the generate matrix is connected."""
    J = random_interaction_matrix(N=100, avk=3, mu=0.8, connected=True)
    n, _ = connected_components(np.abs(J))
    assert n == 1
