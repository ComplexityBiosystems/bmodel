"""
Base classes.

Francesc Font-Clos
Oct 2018
"""
import numpy as np
import pandas as pd

from .utils import check_interaction_matrix
from .io import topo2interaction
from .rules import majority
from .rules import majority_fast


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
        self.node_labels = np.array(node_labels)

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
            columns=["switched_node", "switched_to", "hold"])

    @staticmethod
    def from_topo(topo_file, maxT=1000):
        J, node_labels = topo2interaction(path=topo_file)
        return Bmodel(J=J, node_labels=node_labels, maxT=maxT)

    def runs(self, n_runs=100, fast=False):
        """Find many steady states starting from random initial conditions."""
        # set run function depending on fast option
        if fast:
            run_function = majority_fast
        else:
            run_function = majority

        new_ss = []
        new_energies = []
        for _ in range(int(n_runs)):
            convergence, s, H, UH, ic = self._run(run_function=run_function)
            if convergence:
                # these two are pd.Series so we better
                # store them in a temporal list and
                # append at the end
                new_ss.append(s)
                if H is not None:
                    new_energies.append(H[-1])
                else:
                    new_energies.append(None)
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

    def get_perturbations(self, switched_node=None, switch_to=None, hold=None, return_ic=False):
        """
        Retrieve perturbations already simulated.

        The function returns the reached steady state from all perturbations
        that switched a node ON (OFF) starting with that node OFF (ON)

        Parameters
        ----------
        switched_node: str
            Label of node that was switched.
        switch_to: str, "OFF" or "ON"
            Direciton to which it was switched.
        hold: bool
            Was the node holded or not during relaxation.
        return_ic: bool
            If True, return also the initial conditions.
            Defaults to False

        """
        assert switch_to in ["ON", "OFF"]
        assert hold in [True, False]
        anti_switch_to_int = {"ON": -1, "OFF": 1}[switch_to]
        idx = \
            (self._perturbations_meta.switched_node == switched_node) &\
            (self._perturbations_meta.switched_to == switch_to) &\
            (self._perturbations_ic[switched_node] == anti_switch_to_int) &\
            (self._perturbations_meta.hold == hold)
        if not return_ic:
            return self._perturbations_ss.loc[idx]
        else:
            return (self._perturbations_ss.loc[idx],
                    self._perturbations_ic.loc[idx])

    def perturbe(
        self,
        initial_condition=None,
        node_to_switch=None,
        switch_to=None,
        hold=True,
        n_runs=100
    ):
        """
        Perturbe a steady states by switching a node.

        Parameters
        ----------
        node_to_switch: str
            Label of the node switched.

        """
        ic = np.array(initial_condition).astype(float).copy()
        # check dimensions
        assert len(ic.shape) == 1
        assert len(ic) == self.N
        # check that we are switching to ON or OFF
        assert switch_to in ["ON", "OFF"]
        # check that the state is steady
        assert np.all((ic - np.sign(self.J_pseudo@ic)) == 0)

        # do the switch
        idx = np.argmax(self.node_labels == node_to_switch)
        if switch_to == "ON":
            ic[idx] = 1
        elif switch_to == "OFF":
            ic[idx] = -1
        else:
            raise RuntimeError("Value of 'switch_to' not recognized.")

        # find wich nodes can be updated
        if hold:
            can_be_updated = np.array([
                i
                for i, x in enumerate(self.node_labels)
                if x != node_to_switch
            ])
            assert idx not in can_be_updated
        else:
            can_be_updated = np.array([
                i
                for i, x in enumerate(self.node_labels)
            ])
        # lists to hold data
        initial_conditions = []
        steady_states = []
        metadata = []
        for _ in range(n_runs):
            _, s, _, _, _ = self._run(
                initial_condition=ic,
                can_be_updated=can_be_updated,
                run_function=majority_fast
            )
            # check that blocked node did not change
            if hold:
                assert s[idx] == ic[idx]
            # convergence does not take into account the blocked node
            convergence = np.all(s[can_be_updated])
            if convergence:
                initial_conditions.append(initial_condition)
                steady_states.append(s)
                metadata.append([node_to_switch, switch_to, hold])

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
            columns=["switched_node", "switched_to", "hold"])
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
             run_function,
             initial_condition=np.empty(0, dtype=np.float64),
             can_be_updated=np.empty(0, dtype=np.int64),
             ):
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
        convergence, s, H, UH, ic = run_function(
            N=self.N,
            J=self.J.astype(float),
            J_pseudo=self.J_pseudo.astype(float),
            maxT=self.maxT,
            initial_condition=initial_condition,
            can_be_updated=can_be_updated
        )
        return convergence, s, H, UH, ic
