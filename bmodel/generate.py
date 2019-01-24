"""
Functions to generate random interaction matrices.

Francesc Font-Clos
Oct 2018
"""
import numpy as np
from scipy.sparse.csgraph import connected_components


def random_interaction_matrix(N=100, avk=3, mu=0.8, connected=True):
    """
    Generate a random interaction matrix.

    Parameters
    ----------
    N: int
        System size.
    avk : float
        Average degree.
    mu: float
        Fraction of positive to non-zero entries.
    connected: bool
        Imposes a network with a single connected component

    Returns
    -------
    J: np.array(N, N)
        Interaction matrix.

    """
    # edge density parameters
    alpha = avk / N  # ratio of non-zero edges
    p = alpha * mu
    q = alpha * (1 - mu)

    # interaction matrix
    J = np.random.choice([1, -1, 0], p=[p, q, 1 - p - q], size=(N, N))
    n, _ = connected_components(np.abs(J))

    # impose one single connected component
    if connected:
        while n != 1:
            J = np.random.choice(
                [1, -1, 0], p=[p, q, 1 - p - q], size=(N, N))
            n, _ = connected_components(np.abs(J))
    return J
