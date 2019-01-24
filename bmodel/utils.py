"""
Utility functions for boolean models.

Francesc Font-Clos
Oct 2018
"""
from typing import Sequence, Any
import numpy as np
import pandas as pd


def state_as_string(state: pd.Series, order: Sequence[Any] = None):
    """Represent a boolean state as a string.

    Transforms -1/1 int representation to 0/1 string representation.


    Parameters
    ----------
    state : pd.Series
        Boolean state in -1,1 representation
    order : Sequence[Any], optional
        How to order the variables. Will take series index if not passed.

    Returns
    -------
    str
        A string of the form '01110010' representing the boolean state.
    """

    if order is None:
        order = state.index
    return "".join(((state+1)/2).astype(int).astype(str).loc[order].values)


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
