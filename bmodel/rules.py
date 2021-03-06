import numpy as np
from numba import jit


@jit(nopython=True)
def majority(
    N: int,
    J: np.array,
    J_pseudo: np.array,
    maxT: int,
    initial_condition=np.empty(0, dtype=np.float64),
    can_be_updated=np.empty(0, dtype=np.int64),
):
    """fast thanks tu numba"""
    if len(can_be_updated) == 0:
        can_be_updated = np.arange(N)
    if len(initial_condition) == 0:
        initial_condition = np.random.choice(
            np.array([-1., 1.]),
            size=N)
    else:
        assert len(initial_condition.shape) == 1
        assert len(initial_condition) == N

    s = initial_condition.copy()
    ic = initial_condition.copy()
    e = -1*s@(J@s)
    H = [e]
    UH = [e]
    convergence = False
    for _ in range(int(maxT)):
        # update rule
        k = np.random.choice(can_be_updated)

        sk_new = np.sign((J_pseudo@s)[k])

        if s[k] != sk_new:
            s[k] = sk_new
            e = -1*s@(J@s)
            H.append(e)
            UH.append(e)
            # check convergence
            if np.all(((s - np.sign(J_pseudo@s)) == 0)[can_be_updated]):
                convergence = True
                return convergence, s, H, UH, ic

        else:
            e = -1*s@(J@s)
            H.append(e)

    return convergence, s, H, UH, ic


@jit(nopython=True)
def majority_fast(
    N: int,
    J: np.array,
    J_pseudo: np.array,
    maxT: int,
    initial_condition=np.empty(0, dtype=np.float64),
    can_be_updated=np.empty(0, dtype=np.int64),
):
    """even faster, because energies are not computed"""
    if len(can_be_updated) == 0:
        can_be_updated = np.arange(N)
    if len(initial_condition) == 0:
        initial_condition = np.random.choice(
            np.array([-1., 1.]),
            size=N)
    else:
        assert len(initial_condition.shape) == 1
        assert len(initial_condition) == N

    s = initial_condition.copy()
    ic = initial_condition.copy()
    convergence = False

    for _ in range(maxT):
        k = np.random.choice(can_be_updated)
        sk_new = np.sign((J_pseudo@s)[k])

        if s[k] != sk_new:
            s[k] = sk_new
            # check convergence
            if np.all(((s - np.sign(J_pseudo@s)) == 0)[can_be_updated]):
                convergence = True
                return convergence, s, None, None, ic
    return convergence, s, None, None, ic
