# Balans: Bandit-based Adaptive Large Neighborhood Search

Balans is a research library written in Python 
for solving Mixed-Integer Programming problems (MIPs) 
using bandit-based adaptive large neighborhood search.

Balans follows a scikit-learn style public interface, 
adheres to [PEP-8 standards](https://www.python.org/dev/peps/pep-0008/), 
and is tested heavily. 

## Quick Start

```python
from alns.accept import HillClimbing
from alns.select import MABSelector
from alns.stop import MaxIterations
from mabwiser.mab import LearningPolicy
from balans.destroy import DestroyOperators
from balans.repair import RepairOperators
from balans.solver import Balans

# Solver
balans = Balans(destroy_ops=[DestroyOperators.Mutation], 
                repair_ops=[RepairOperators.Repair], 
                selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=5, num_repair=1,
                                       learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15)),
                accept=HillClimbing(),
                stop=MaxIterations(5))

# Run
result = balans.solve("neos-5140963-mincio.mps.gz")

# Result
print("Best solution:", result.best_state.objective())
```

## Available Destroy Operators
* Crossover [reference]
* Dins
* Local_Branching
* Mutation
* No_Objective
* Proximit
* Rens

## Available Repair Operators
* Repair Sub-MIP


## Installation

1. Install a Python-compatible[^1] version of [SCIP Optimization Solver](https://www.scipopt.org/index.php#download) which requires prepackaged C++ libraries[^2]. 
[^1]: The Pyhon interface only works with major solver versions, see [(SCIP Compatibility Table](https://pypi.org/project/PySCIPOpt/) to pick the right solver version.
[^2]: The Solver is written in C++ so it requires [Visual C++ Redistributable Packages](), see the link under precompiled packages section.
Alternatively, here is the official [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to download and install C++ development tools. 
2. Now that the Solver and the required C++ backend is installed, set environment variable for [SCIPOPTDIR](https://imada.sdu.dk/u/marco/DM871/PySCIPOpt/md_INSTALL.html)
3. Now that the Solver, required C++ backend is installed and is references in the environment, install the Python interface v`pip install pyscipopt` via pip[^3]
[^3]: A good practice is to update first via `pip install --upgrade` or `python -m pip install --upgrade pip`

Balans can be installed from the wheel file or building from source by following the instructions in 
our [documentation](https://github.io/balans/installation.html).

## License

Balans is licensed under the [Apache License 2.0](LICENSE.md).

## Installation
Balans requires Python 3.8+ and can be installed from the provided wheel file.  

```
$ git clone https://github.com/skadio/balans   
$ cd xxx
$ pip install dist/xxx-X.X.X-py3-none-any.whl
```

## Running Unit Tests

```
$ cd balans
$ python -m unittest discover tests
```

## Changelog

| Date | Notes |
|--------|-------------|
| July 7, 2023 | Initial release |

## References

1. xxx
2. xx

<br>
