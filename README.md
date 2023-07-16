# Balans: Bandit-based Adaptive Large Neighborhood Search

Balans is a research library written in Python 
for solving Mixed-Integer Programming problems (MIPs) 
using bandit-based adaptive large neighborhood search.

Balans integrates [MABWiser](https://github.com/fidelity/mabwiser/) for multi-armed bandits,
[ALNS](https://github.com/N-Wouda/ALNS/) for adaptive large neighborhood search, and 
[SCIP](https://scipopt.org/) for solving mixed-integer linear programming problems. 

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
* Crossover[^1]
[^1]: xxx
* Dins[^2]
[^2]: xxx
* Local_Branching[^3]
[^3]: xxx
* Mutation[^4]
[^4]: xxx
* No Objective[^5]
[^5]: xxx 
* Proximity[^6]
[^6]: xxx
* Rens[^7]
[^7]: xxx

## Available Repair Operators
* Repair MIP

## Installation

Balans requires Python 3.8+ and SCIP Solver and can be installed from PyPI via `pip install balans`. 

### Dependencies 

Balans depends on [MABWiser](https://github.com/fidelity/mabwiser/) for multi-armed bandits,
[ALNS](https://github.com/N-Wouda/ALNS/) for adaptive large neighborhood search, and 
[SCIP](https://scipopt.org/) for solving mixed-integer linear programming problems. 

While MABWiser and ALNS are pip-installable as shown in [requirements.txt](https://github.com/skadio/balans/blob/main/requirements.txt), SCIP needs to be installed: 

1. Install a Python-compatible[^8] version of [SCIP Optimization Solver](https://www.scipopt.org/index.php#download) which requires prepackaged C++ libraries[^9].
[^8]: The Pyhon interface only works with major solver versions, see [SCIP Compatibility Table](https://pypi.org/project/PySCIPOpt/) to pick the right solver version.
[^9]: The Solver is written in C++ so it requires [Visual C++ Redistributable Packages](), see the link under precompiled packages section. Alternatively, here is the official [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to download and install C++ development tools. 
2. Now that the Solver _and_ the required C++ backend are installed, set the environment variable for [SCIPOPTDIR](https://imada.sdu.dk/u/marco/DM871/PySCIPOpt/md_INSTALL.html).
3. Now that the Solver and required C++ backend are installed _and_ is references in the environment, install the Python interface via `pip install pyscipopt` or `conda install --channel conda-forge pyscipopt`[^10].
[^10]: A good practice is to update first via `pip install --upgrade` or `python -m pip install --upgrade pip`

## Test Your Setup

```
$ cd balans
$ python -m unittest discover tests
```

## Changelog

| Date | Notes |
|--------|-------------|
| July 7, 2023 | Initial release |



## License

Balans is licensed under the [Apache License 2.0](LICENSE.md).


## References

1. xxx
2. xx

<br>
