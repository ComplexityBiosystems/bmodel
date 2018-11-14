"""
Base classes.

Francesc Font-Clos
Oct 2018
"""
import numpy as np
import pandas as pd
from .utils import check_interaction_matrix


class Bmodel():
    """Main class holding a Boolean Model."""

    def __init__(self, maxT=1000, J=None, node_labels=None):
        """
        Create a boolean model instance.

        Notes
        -----
        I define the pseudo-interaction_matrix
        The matrix J_pseudo is such that sign(J_pseudo.dot(s)) is like
        sign(J.dot(s))+taking_ties_into_account

        """
        # check input
        assert J is not None
        num_nodes, M = J.shape
        assert num_nodes == M
        assert check_interaction_matrix(J)
        # pseudo-interaction matrix
        J_pseudo = np.identity(num_nodes) + 2 * J

        # deal with named nodes
        if node_labels is None:
            node_labels = range(num_nodes)
        self.node_labels = node_labels

        # store stuff
        self.J = J
        self.J_pseudo = J_pseudo
        self.N = num_nodes
        self.maxT = maxT
        self.total_runs = 0
        self.steady_states = pd.DataFrame(columns=node_labels, dtype=int)
        self.energies = pd.Series(dtype=int)
        self._energy_paths = []
        self._unique_energy_paths = []
        self._initial_conditions = []
        self._perturbations_ic = pd.DataFrame(columns=node_labels, dtype=int)
        self._perturbations_ss = pd.DataFrame(columns=node_labels, dtype=int)
        self._perturbations_meta = pd.DataFrame(
            columns=["switched_node", "switched_to"])

    def runs(self, n_runs=100):
        """Find many steady states starting from random initial conditions."""
        new_ss = []
        new_energies = []
        for _ in range(int(n_runs)):
            convergence, s, H, UH, ic = self._run()
            if convergence:
                # these two are pd.Series so we better
                # store them in a temporal list and
                # append at the end
                new_ss.append(s)
                new_energies.append(H[-1])
                # these two are lists so it doesn't hurt to append now
                self._energy_paths.append(H)
                self._unique_energy_paths.append(UH)
                self._initial_conditions.append(ic)

        self.steady_states = self.steady_states.append(
            pd.DataFrame(new_ss, columns=self.node_labels),
            ignore_index=True)
        self.energies = self.energies.append(
            pd.Series(new_energies),
            ignore_index=True)

        # to keep track of convergence probability
        self.total_runs += n_runs

    def get_perturbations(self, switched_node=None, switch_to=None):
        """
        Retrieve perturbations already simulated.

        The function returns the reached steady state from all perturbations
        that switched a node ON (OFF) starting with that node OFF (ON)

        Parameters
        ----------
        switched_node: str
            Label of node that was switched.
        switch_to: "OFF" or "ON"
            Direciton to which it was switched.

        """
        idx = \
            (self._perturbations_meta.switched_node == switched_node) &\
            (self._perturbations_meta.switched_to == "ON") &\
            (self._perturbations_ic[switched_node] == -1)
        return self._perturbations_ss.loc[idx]

    def perturbe(
        self,
        initial_condition=None,
        node_to_switch=None,
        switch_to=None,
        n_runs=100
    ):
        """
        Perturbe a steady states by switching a node.

        Parameters
        ----------
        node_to_switch: str
            Label of the node switched.

        """
        ic = np.array(initial_condition).copy()
        # check dimensions
        assert len(ic.shape) == 1
        assert len(ic) == self.N
        # check that we are switching to ON or OFF
        assert switch_to in ["ON", "OFF"]
        # check that the state is steady
        assert np.all((ic - np.sign(self.J_pseudo@ic)) == 0)

        # do the switch
        i = np.argmax(self.node_labels == node_to_switch)
        if switch_to == "ON":
            ic[i] = 1
        elif switch_to == "OFF":
            ic[i] = -1
        else:
            raise RuntimeError("Value of 'switch_to' not recognized.")

        # find wich nodes can be updated
        can_be_updated = [i for i, x in enumerate(self.node_labels)
                          if x != node_to_switch]
        # lists to hold data
        initial_conditions = []
        steady_states = []
        metadata = []
        for _ in range(n_runs):
            _, s, H, UH, _ = self._run(
                initial_condition=ic,
                can_be_updated=can_be_updated
            )
            # check that blocked node did not change
            assert s[i] == ic[i]
            # convergence does not take into account the blocked node
            would_not_change = s == np.sign(self.J_pseudo@s)
            convergence = np.all(would_not_change[can_be_updated])
            if convergence:
                initial_conditions.append(initial_condition)
                steady_states.append(s)
                metadata.append([node_to_switch, switch_to])

        # add new gathered perturbation data to bmodel class
        new_perturbations_ic = pd.DataFrame(
            initial_conditions,
            columns=self.node_labels,
            dtype=int)
        new_perturbations_ss = pd.DataFrame(
            steady_states,
            columns=self.node_labels,
            dtype=int)
        new_perturbations_meta = pd.DataFrame(
            metadata,
            columns=["switched_node", "switched_to"])
        self._perturbations_ic = self._perturbations_ic.append(
            new_perturbations_ic,
            ignore_index=True)
        self._perturbations_ss = self._perturbations_ss.append(
            new_perturbations_ss,
            ignore_index=True)
        self._perturbations_meta = self._perturbations_meta.append(
            new_perturbations_meta,
            ignore_index=True)

    def _run(self,
             initial_condition=None,
             can_be_updated=None):
        """
        Find one steady state.

        Parameters
        ----------
        initial_condition: np.array(N,)
            Initial condition. Defaults to random.
        can_be_updated: list-like
            Nodes that can be updated. OE/KD nodes cannot be updated.

        return convergence, s, H, UH

        """
        if initial_condition is None:
            initial_condition = np.random.choice([-1, 1], size=(self.N))
        else:
            assert len(initial_condition.shape) == 1
            assert len(initial_condition) == self.N
        if can_be_updated is None:
            can_be_updated = range(self.N)
        s = initial_condition
        ic = s.copy()
        e = -s@(self.J@s)
        H = [e]
        UH = [e]
        convergence = False

        for _ in range(int(self.maxT)):
            # update rule
            k = np.random.choice(can_be_updated)

            sk_new = np.sign((self.J_pseudo@s)[k])

            if s[k] != sk_new:
                s[k] = sk_new
                e = -s@(self.J@s)
                H.append(e)
                UH.append(e)
                # check convergence
                if np.all(((s - np.sign(self.J_pseudo@s)) == 0)[can_be_updated]):
                    convergence = True
                    break
            else:
                e = -s@(self.J@s)
                H.append(e)

        return convergence, s, H, UH, ic
