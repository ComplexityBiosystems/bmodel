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

## Basic Usage
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
bmodel.steady_states
```

When the option `fast` is set to `False`, the code is slower but stores more data. For instance, the *energy* of the steady states has already been calculated and is stored as an attribute

```python
bmodel.energies
```
If instead we set the `fast` option to `True`, the code is much faster but stores only the steady states. This can be useful when exploring large models that have a very large number of steady states. 

```python
bmodel.runs(n_runs=1000, fast=True)
```

Notice that calling `.runs()` a second time will append to the steady states you already found!

## Using pathways from the bmodel library
The `bmodel` package comes with a set of ready-to-use EMT-related pathways, which can be loaded with a one-liner. For instance
```python
# load the EMT-MET pathway from (Font-Clos et al, 2018)
bmodel = Bmodel.from_library("EMT_MET")
```
loads the EMT-MET model used in (Font-Clos et al, 2018). The rest of pathways are smaller in size and some of them have been studied in (Kishore et al, 2019):
| Name         | Number of nodes | Number of edges |
| ------------ | --------------- | --------------- |
| EMT_MET      | 72              | 142             |
| EMT_RACIPE   | 22              | 82              |
| repressors_5 | 10              | 18              |
| OVOL2_CBS    | 9               | 20              |
| OVOL2_Jia    | 9               | 17              |
| NRF2         | 8               | 16              |
| OVOL2_Jia_2  | 8               | 15              |
| NP63         | 6               | 9               |
| EMT_core     | 6               | 12              |
| miR145OCT4   | 5               | 10              |
| OVOL2        | 4               | 9               |
| GRHL2        | 4               | 7               |

Thus if you want to load a smaller model such as *NP63* which has only 6 nodes and 9 interactions, you can simply do:

```python
# load the EMT-MET pathway from (Font-Clos et al, 2018)
bmodel = Bmodel.from_library("NP63")
```


## Advanced Usage
If you want to work with your own models, you might want to load them from a text file. As of today, `bmodel` can read `.topo` files and more generic `.csv` files containing the list of interactions. The syntax for those is similar:

```python
# load from .topo file
bmodel = Bmodel.from_topo("path/to/file.topo")

# load from .csv file
bmodel = Bmodel.from_edgelist("path/to/file.csv")
```
and the expected formatting of the text files is specified in the `topo2interaction` and `edgelist2interaction` docstrings of the [io](bmodel/io.py) submodule.

