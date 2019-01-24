"""
Base classes.

Francesc Font-Clos
Oct 2018
"""
import numpy as np
import pandas as pd
from typing import Sequence, List

from .utils import check_interaction_matrix
from .io import topo2interaction
from .rules import majority
from .rules import majority_fast
from .exceptions import IndicatorError

"""
    Create a boolean model instance.


    """


class Bmodel():
    """Main class holding a Boolean Model."""

    def __init__(
        self,
        J: np.ndarray = None,
        node_labels: Sequence[str] = None,
        indicator_nodes: Sequence[str] = [],
        maxT=1000,
    ):
        r"""Boolean model with majority rule dynamics.

        Parameters
        ----------
        J : np.ndarray, optional
            Interaction matrix (the default is None, which [default_description])
        node_labels : Sequence, optional
            Labels given to the nodes. If not passed, integer labels are used.
        indicator_nodes : Sequence, optional
            Nodes that indicate a phenotypic state but don't represent a real
            entity in the model. E.g. "Apoptosis"
        maxT : int, optional
            Maximum number of time-steps to find steady states. Defaults to 1000.

        Notes
        -----
        The pseudo-interaction matrix J_pseudo is such that

        `sign(J_pseudo.dot(s))`

        is equivalent to

        `sign(J.dot(s)) + taking_ties_into_account`

        What should indicator nodes behave like?
        1. Cannot be input of other nodes
        2. Cannot be blocked
        3. Should be neutral (0) in initial conditions

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
            node_labels = ["node_" + str(x) for x in range(num_nodes)]
        self.node_labels = np.array(node_labels)

        # store stuff
        self.J = J
        # set indicator nodes
        self._set_indicator_nodes(indicator_nodes)
        self.J_pseudo = J_pseudo
        self.N = num_nodes
        self.maxT = maxT
        self.total_runs = 0
        self.steady_states = pd.DataFrame(columns=node_labels, dtype=int)
        self.energies = pd.Series(dtype=int)
        self._energy_paths: List = []
        self._unique_energy_paths: List = []
        self._initial_conditions: List = []
        self._perturbations_ic = pd.DataFrame(columns=node_labels, dtype=int)
        self._perturbations_ss = pd.DataFrame(columns=node_labels, dtype=int)
        self._perturbations_meta = pd.DataFrame(
            columns=["switched_node", "switched_to", "hold", "label"])

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
            initial_condition = np.empty(0, dtype=np.float64)
            # set neutral value 0 to indicator nodes
            if self._indicator_idx:
                initial_condition = np.random.choice(
                    np.array([-1., 1.]),
                    size=self.N
                )
                initial_condition[self._indicator_idx] = 0
            convergence, s, H, UH, ic = self._run(
                initial_condition=initial_condition,
                run_function=run_function
            )
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
        label="",
        node_to_switch=None,
        switch_to=None,
        hold=None,
        allow_non_steady_state=False,
        n_runs=100
    ):
        """
        Perturbe a steady states by switching a node.

        Parameters
        ----------
        node_to_switch: str
            Label of the node switched.
        label: str
            Label with soem information about the initial condition
        """
        assert initial_condition is not None
        assert node_to_switch is not None
        assert node_to_switch not in self.indicator_nodes
        assert switch_to is not None
        assert hold in [False, True]

        # parse the initial condition
        ic = self._parse_initial_condition(initial_condition)
        # fill in the indicator nodes
        ic = self._fill_in_indicator_nodes(ic)

        # check that we are switching to ON or OFF
        assert switch_to in ["ON", "OFF"]
        # check that the state is steady
        if not allow_non_steady_state:
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
                metadata.append([node_to_switch, switch_to, hold, label])

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
            columns=["switched_node", "switched_to", "hold", "label"])
        self._perturbations_ic = self._perturbations_ic.append(
            new_perturbations_ic,
            ignore_index=True)
        self._perturbations_ss = self._perturbations_ss.append(
            new_perturbations_ss,
            ignore_index=True)
        self._perturbations_meta = self._perturbations_meta.append(
            new_perturbations_meta,
            ignore_index=True)

    # ----------------
    # PRIVATE METHODS
    # ----------------

    def _set_indicator_nodes(self, indicator_nodes: Sequence) -> None:
        # check that we have unique elements
        if not sorted(set(list(indicator_nodes))) == sorted(list(indicator_nodes)):
            raise IndicatorError

        # check indicator nodes exist
        for node in indicator_nodes:
            if node not in self.node_labels:
                raise IndicatorError(
                    f"Label {node} not found in node_labels, so it cannot be set as indicator node")
        self.indicator_nodes = list(indicator_nodes)
        self._indicator_idx = [
            i
            for i, x in enumerate(self.node_labels)
            if x in self.indicator_nodes
        ]
        # check that indicator nodes are not input of other nodes
        for i in self._indicator_idx:
            if np.any(self.J[:, i] != 0):
                raise IndicatorError(
                    f"Node {self.node_labels[i]} cannot be an indicator because it is input of other nodes")

    def _run(
        self,
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

    def _fill_in_indicator_nodes(
        self,
        initial_condition=np.empty(0, dtype=np.float64),
        can_be_updated=np.empty(0, dtype=np.int64),
    ):
        s = initial_condition
        # for each indicator node, update only him
        for i in self._indicator_idx:
            can_be_updated = np.array([i])
            _, s, _, _, _ = majority_fast(
                N=self.N,
                J=self.J.astype(float),
                J_pseudo=self.J_pseudo.astype(float),
                maxT=1,
                initial_condition=s,
                can_be_updated=can_be_updated
            )
        return s

    def _parse_initial_condition(self, initial_condition):
        assert isinstance(initial_condition,
                          (tuple, list, np.ndarray, dict, pd.Series))

        # if type is list or array, length must be correct
        if isinstance(initial_condition, (tuple, list, np.ndarray)):
            assert len(initial_condition) == self.N
            ic = np.array(initial_condition).astype(float)
            # make sure indicator nodes are all 0
            if not np.all(ic[self._indicator_idx] == 0):
                raise IndicatorError(
                    "The initial condition contains non-zero elements at positions that correspond to indicator nodes.")

        # in the other cases, we have named entries
        # either we get all the nodes, or we miss all indicator nodes
        elif isinstance(initial_condition, dict):
            ic = []
            for node in self.node_labels:
                if node not in initial_condition.keys():
                    # if we are missing a node we check if
                    # its an indicator node because that's ok
                    if node in self.indicator_nodes:
                        ic.append(0)
                        # but then we check that none other indicator nodes
                        # have been passsed as neither
                        for _node in self.indicator_nodes:
                            if _node != node:
                                if _node in initial_condition.keys():
                                    raise IndicatorError(
                                        f"Indicator node {node} missing but {_node} present")
                    else:
                        raise RuntimeError(f"Node {node} not in keys")
                else:
                    value = initial_condition[node]
                    ic.append(value)
        # if its a Series revert to dict
        elif isinstance(initial_condition, pd.Series):
            return self._parse_initial_condition(dict(initial_condition))

        # finally, check that we got the right values
        for idx, val in enumerate(ic):
            if idx in self._indicator_idx:
                assert val == 0
            else:
                if val not in [-1, 1]:
                    raise RuntimeError(
                        f"All values must be in [-1, 1] (for node {self.node_labels[idx]} you passed {val})")

        return np.array(ic).astype(float)
