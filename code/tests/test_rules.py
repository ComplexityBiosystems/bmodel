from bmodel.rules import majority
from bmodel.rules import majority_fast

import pytest
import numpy as np


@pytest.fixture
def neg_feedback():
    N = 2
    J = np.array([[0, -1], [-1, 0]])
    J_pseudo = np.identity(N) + 2 * J
    return N, J, J_pseudo


def test_majority_fast_return_types(neg_feedback):
    N, J, J_pseudo = neg_feedback
    maxT = 100
    initial_condition = np.array([1., -1.])
    convergence, s, none1, none2, ic = majority_fast(
        N=N,
        J=J,
        J_pseudo=J_pseudo,
        maxT=maxT,
        initial_condition=initial_condition,
    )
    assert isinstance(convergence, type(True))
    assert isinstance(s, np.ndarray)
    assert isinstance(none1, type(None))
    assert isinstance(none2, type(None))
    assert isinstance(ic, np.ndarray)
