"""
Functions to handle input/ouput.

Francesc Font-Clos
Oct 2018
"""
import os
import pandas as pd
import numpy as np


def topo2interaction(path):
    """
    Construct interaction matrix from .topo file.

    Parameters
    ----------
    path: str
        Path to .topo file.

    Returns
    -------
    J: np.array(N, N)
        Interaction matrix

    """
    # check topo file is fine
    assert os.path.isfile(path)
    table = pd.read_csv(path, sep=" ")
    assert np.all(table.columns == np.array(["Source", "Target", "Type"]))

    # deal with node labels
    tmp = table[["Source", "Target"]].values.reshape(-1)
    node_labels = sorted(list(set(tmp)))
    N = len(node_labels)
    idx2label = dict(enumerate(node_labels))
    label2idx = {v: k for k, v in idx2label.items()}

    # dict to convert topo to J representation
    T2J_dict = {1: 1, 2: -1}

    # compose interaction matrix J
    J = np.zeros((N, N), int)
    for u, v, t in table.values:
        # Warning: J_{ij} is the interaction from j to i
        # J_{ij} = J_{i <- j}
        j = label2idx[u]
        i = label2idx[v]
        J[i, j] = T2J_dict[t]

    return J, np.array(node_labels)


def edgelist2interaction(
        path,
        header=["from", "to", "type"],
        pos_symb=1,
        neg_symb=-1):
    """Construct interaction matrix from edgelist csv file

    Parameters
    ----------
    path : str
        Path to csv file
    header : list, optional
        First line of csv file, by default ["from", "to", "type"]
    pos_symb : int, optional
        Symbol used to represent positive interactions in csv file, by default 1
    neg_symb : int, optional
        Symbol used to represent negative interactions in csv file, by default -1

    Returns
    -------
    J : np.ndarray
        Interaction matrix
    node_labels: np.ndarray
        Node labels
    """
    # check csv file is fine
    assert os.path.isfile(path)
    table = pd.read_csv(path)
    assert np.all(table.columns == np.array(header))

    # deal with node labels
    from_field, to_field, _ = header
    tmp = table[[from_field, to_field]].values.reshape(-1)
    node_labels = sorted(list(set(tmp)))
    N = len(node_labels)
    idx2label = dict(enumerate(node_labels))
    label2idx = {v: k for k, v in idx2label.items()}

    # dict to convert topo to J representation
    T2J_dict = {pos_symb: 1, neg_symb: -1}

    # compose interaction matrix J
    J = np.zeros((N, N), int)
    for u, v, t in table.values:
        # Warning: J_{ij} is the interaction from j to i
        # J_{ij} = J_{i <- j}
        j = label2idx[u]
        i = label2idx[v]
        J[i, j] = T2J_dict[t]

    return J, np.array(node_labels)
