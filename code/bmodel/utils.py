"""
Utility functions for boolean models.

Francesc Font-Clos
Oct 2018
"""
import numpy as np


def check_interaction_matrix(J):
    """
    Check the interaction matrix.

    This function checks the following:
    1. That the matrix is composed only of -1, 0, 1.
    2. That it is not composed only of 0's
    3. That it is a np.array
    """
    # check 3
    if not isinstance(J, np.ndarray):
        return False

    vals = sorted(list(set(J.reshape(-1))))

    # check 1
    for val in vals:
        if val not in {-1, 0, 1}:
            return False

    # check 2
    if len(vals) == 1 and vals[0] == 0:
        return False

    # all good
    return True
