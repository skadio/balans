# Balans: Bandit-based Adaptive Large Neighborhood Search

Balans is a research library written in Python to serve as a meta-solver 
for Mixed-Integer Programming problems (MIPs) using 
multi-armed bandit-based adaptive large neighborhood search.

The framework integrates [MABWiser](https://github.com/fidelity/mabwiser/) for contextual multi-armed bandits,
[ALNS](https://github.com/N-Wouda/ALNS/) for adaptive large neighborhood search, and 
[SCIP](https://scipopt.org/) for solving mixed-integer linear programming problems. 

## Quick Start

```python
# Adaptive large neigborhood
from alns.select import MABSelector
from alns.accept import HillClimbing
from alns.stop import MaxIterations

# Contextual multi-armed bandits
from mabwiser.mab import LearningPolicy

# Meta-solver built on top of SCIP
from balans.solver import Balans, DestroyOperators, RepairOperators

# Balans
balans = Balans(destroy_ops=[DestroyOperators.Crossover,
                             DestroyOperators.Dins, 
                             DestroyOperators.Mutation, 
                             DestroyOperators.Local_Branching,
                             DestroyOperators.Proximity,
                             DestroyOperators.Rens, 
                             DestroyOperators.Rins,
                             DestroyOperators.Crossover,
                             DestroyOperators.Proximity,
                             DestroyOperators.Zero_Objective],
                repair_ops=[RepairOperators.Repair],
                selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=8, num_repair=1,
                                     learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15)),
                accept=HillClimbing(),
                stop=MaxIterations(100))

# Run
result = balans.solve("noswot.mps")

# Result
print("Best solution:", result.best_state.solution())
print("Best solution objective:", result.best_state.objective())
```

## Available Destroy Operators
* Dins[^1] 
[^1]: S. Ghosh. DINS, a MIP Improvement Heuristic. Integer Programming and Combinatorial Optimization: IPCO, 2007.
* Local Branching[^2]
[^2]: M. Fischetti and A. Lodi. Local branching. Mathematical Programming, 2003.
* Mutation[^3]
[^3]: Rothberg. An Evolutionary Algorithm for Polishing Mixed Integer Programming Solutions. INFORMS Journal on Computing, 2007.
* Rens[^4]
[^4]: Berthold. RENS–the optimal rounding. Mathematical Programming Computation, 2014.
* Rins[^5]
[^5]: E. Danna, E. Rothberg, and C. L. Pape. Exploring relaxation induced neighborhoods to improve MIP solutions. Mathematical Programming, 2005.
* Zero Objective[^6]
[^6]: Zero Objective.
* Proximity Search[^7]
[^7]: M. Fischetti and M. Monaci. Proximity search for 0-1 mixed-integer convex programming. Journal of Heuristics, 20(6):709–731, Dec 2014.
* Crossover[^8]
[^8]: E. Rothberg. An Evolutionary Algorithm for Polishing Mixed Integer Programming Solutions. INFORMS Journal on Computing, 19(4):534–541, 2007.

## Available Repair Operators
* Repair MIP

## Installation

Balans requires Python 3.8+ and SCIP Solver and can be installed from PyPI via `pip install balans`. 

### Dependencies 

Balans depends on [MABWiser](https://github.com/fidelity/mabwiser/) for multi-armed bandits,
[ALNS](https://github.com/N-Wouda/ALNS/) for adaptive large neighborhood search, and 
[SCIP](https://scipopt.org/) for solving mixed-integer linear programming problems. 
While MABWiser and ALNS are pip-installable as shown in [requirements.txt](https://github.com/skadio/balans/blob/main/requirements.txt), 
SCIP needs to be installed: 

1. Install a Python-compatible[^7] version of [SCIP Optimization Solver](https://www.scipopt.org/index.php#download) which requires C++ prepackage libraries[^8].
[^7]: The Python interface of SCIP only works with major versions, see [SCIP Compatibility Table](https://pypi.org/project/PySCIPOpt/) to pick the right solver version.
[^8]: SCIP is written in C++ so it requires [Visual C++ Redistributable Packages](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170), check the link under precompiled packages section. Alternatively, here is the official [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to download and install C++ development tools (make sure to click on C++ tools in the installer). 
2. Now that SCIP _and_ the required C++ backend are installed, set the environment variable for [SCIPOPTDIR](https://imada.sdu.dk/u/marco/DM871/PySCIPOpt/md_INSTALL.html).
3. Now that SCIP and required C++ backend are installed _and_ is references in the environment, install the Python interface via `pip install pyscipopt` or `conda install --channel conda-forge pyscipopt`[^9].
[^9]: A good practice is to update first via `pip install --upgrade` or `python -m pip install --upgrade pip`

## Test Your Setup

```
$ cd balans
$ python -m unittest discover tests
```

## License

Balans is licensed under the [Apache License 2.0](LICENSE).


<br>
