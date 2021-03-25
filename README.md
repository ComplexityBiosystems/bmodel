# bmodel
Boolean model simulation in Python 3. This code is used to generate Boolean model simulations of regulatory pathways including the EMT.

## Publications
This repository or parts of it have been used in the following publications so far:

1. Font-Clos F, Zapperi S, and La Porta CAM. 2018. “Topography of Epithelial-Mesenchymal Plasticity.” Proceedings of the National Academy of Sciences of the United States of America 115 (23): 5902–7. https://doi.org/10.1073/pnas.1722609115.
2. Kishore H, Sabuwala B, Subramani BV, La Porta CAM, Zapperi S, Font-Clos F, and Jolly MK. 2019. “Identifying Inhibitors of Epithelial-Mesenchymal Plasticity Using a Network Topology Based Approach.” bioRxiv. https://doi.org/10.1101/854307. 
3. Font-Clos F, Zapperi S, and La Porta CAM. Classification of triple-negative breast cancers through a Boolean network model of the epithelial-mesenchymal transition.
submitted (2020).

## Installation
Clone and install locally via pip in editable mode as follows
```bash
git clone https://github.com/ComplexityBiosystems/bmodel.git
cd bmodel
pip install -e .
```

## Tests
To make sure the package runs correctly in your system, run
```bash
pytest -v
```

## Usage
To create a simple Boolean model we just need an interaction matrix $J$ and some labels for the nodes. The base `Bmodel` class is instantiated as follows:
```python
import numpy as np
from bmodel.base import Bmodel

# create a simple interaction matrix
J = np.array([
       [ 0,  0, -1,  0],
       [ 0,  0,  0,  0],
       [-1,  1,  1, -1],
       [ 0, -1, -1,  0]])
])

# give names to the nodes
node_labels = ["A", "B", "C", "D"]

# instantiate a Boolean model
bmodel = Bmodel(J=J, node_labels=node_labels)
```
We can now run some simulations with a single command, for instance
```python
bmodel.runs(n_runs=100, fast=False)
```
will run 100 simulations, starting with random initial conditions. The reached steady states are conveniently stored in a dataframe as an attribute:
```python
bmodel.stead_states
```

When the option `fast` is set to `False`, the code is slower but stores more data. For instance, the *energy* of the steady states has already been calculated and is stored as an attribute

```python
bmodel.energies
```
If instead we set the `fast` option to `True`, the code is much faster but stores only the steady states. This can be useful when exploring large models that have a very large number of steady states. 

