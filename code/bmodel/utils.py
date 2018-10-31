"""
Utility functions for boolean models.

Francesc Font-Clos
Oct 2018
"""


def check_interaction_matrix(J):
    """
    Check the interaction matrix.

    Check that the interaction matrix is composed only
    of -1, 0, 1.
    """
    vals = sorted(list(set(J.reshape(-1))))
    for val in vals:
        assert val in {-1, 0, 1}
    return True
