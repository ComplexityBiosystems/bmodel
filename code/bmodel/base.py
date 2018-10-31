"""
Base classes.

Francesc Font-Clos
Oct 2018
"""
import numpy as np
import pandas as pd
from .utils import check_interaction_matrix


class Bmodel(object):
    """Main class holding a Boolean Model."""

    def __init__(self, maxT=1000, J=None, node_labels=None):
        """
        Create a boolean model instance.

        The matrix A is such that sign(A.dot(s)) is like
        sign(J.dot(s))+taking_ties_into_account

        """
        # check input
        assert J is not None
        N, M = J.shape
        assert N == M
        assert check_interaction_matrix(J)
        # pseudo-interaction matrix
        A = np.identity(N) + 2 * J

        # deal with named nodes
        if node_labels is None:
            node_labels = range(N)
        self.node_labels = node_labels

        # store stuff
        self.J = J
        self.A = A
        self.N = N
        self.maxT = maxT
        self.total_runs = 0
        self.ss = pd.DataFrame(columns=node_labels, dtype=int)
        self.energies = pd.Series(dtype=int)
        self.H_paths = []
        self.UH_paths = []
        self.initial_conditions = []

    def runs(self, n_runs=100):
        """Find many steady states starting from random initial conditions."""
        new_ss = []
        new_energies = []
        for _ in range(int(n_runs)):
            convergence, s, H, UH = self._run()
            if convergence:
                # these two are pd.Series so we better
                # store them in a temporal list and
                # append at the end
                new_ss.append(s)
                new_energies.append(H[-1])
                # these two are lists so it doesn't hurt to append now
                self.H_paths.append(H)
                self.UH_paths.append(UH)

        self.ss = self.ss.append(
            pd.DataFrame(new_ss, columns=self.node_labels),
            ignore_index=True)
        self.energies = self.energies.append(
            pd.Series(new_energies),
            ignore_index=True)

        # to keep track of convergence probability
        self.total_runs += n_runs

    def _run(self):
        """
        Find one steady state starting from random initial conditions.

        return convergence, s, H, UH
        """
        s = np.random.choice([-1, 1], size=(self.N))
        self.initial_conditions.append(s)
        e = -s@(self.J@s)
        H = [e]
        UH = [e]
        convergence = False

        for i in range(int(self.maxT)):
            # update rule
            k = np.random.choice(range(self.N))

            sk_new = np.sign((self.A@s)[k])

            if s[k] != sk_new:
                s[k] = sk_new
                e = -s@(self.J@s)
                H.append(e)
                UH.append(e)
                # check convergence
                if np.all((s - np.sign(self.A@s)) == 0):
                    convergence = True
                    break
            else:
                e = -s@(self.J@s)
                H.append(e)

        return convergence, s, H, UH
