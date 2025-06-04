# Balans: Bandit-based Adaptive Large Neighborhood Search

Balans is a meta-solver for Mixed-Integer Programming problems (MIPs) using 
multi-armed bandit-based adaptive large neighborhood search.

The hybrid framework integrates [MABWiser](https://github.com/fidelity/mabwiser/) for contextual multi-armed bandits,
[ALNS](https://github.com/N-Wouda/ALNS/) for adaptive large neighborhood search, and 
[SCIP](https://scipopt.org/) or [Gurobi](https://www.gurobi.com/) for solving mixed-integer linear programming problems. 

## Quick Start

```python
# ALNS for adaptive large neigborhood search
from alns.select import MABSelector
from alns.accept import HillClimbing, SimulatedAnnealing
from alns.stop import MaxIterations, MaxRuntime

# MABWiser for contextual multi-armed bandits
from mabwiser.mab import LearningPolicy

# Balans meta-solver for solving mixed integer programming problems
from balans.solver import Balans, DestroyOperators, RepairOperators

# Destroy operators
destroy_ops = [DestroyOperators.Crossover,
               DestroyOperators.Dins,
               DestroyOperators.Mutation_25,
               DestroyOperators.Local_Branching_10,
               DestroyOperators.Rins_25,
               DestroyOperators.Proximity_05,
               DestroyOperators.Rens_25,
               DestroyOperators.Random_Objective]

# Repair operators
repair_ops = [RepairOperators.Repair]

# Rewards for online learning feedback loop
best, better, accept, reject = 4, 3, 2, 1

# Bandit selector
selector = MABSelector(scores=[best, better, accept, reject],
                       num_destroy=len(destroy_ops),
                       num_repair=len(repair_ops),
                       learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50))

# Acceptance criterion
# accept = HillClimbing() # for pure exploitation 
accept = SimulatedAnnealing(start_temperature=20, end_temperature=1, step=0.1)

# Stopping condition
# stop = MaxRuntime(100)
stop = MaxIterations(10)

# Balans
balans = Balans(destroy_ops=destroy_ops,
                repair_ops=repair_ops,
                selector=selector,
                accept=accept,
                stop=stop,
                mip_solver="scip", # "gurobi"
                n_threads=1) # gurobi can have multiple threads for parallelization

# Run a mip instance to retrieve results 
instance_path = "data/miplib/noswot.mps"
result = balans.solve(instance_path)

# Results of the best found solution and the objective
print("Best solution:", result.best_state.solution())
print("Best solution objective:", result.best_state.objective())
```

## Parallel Version of Balans

We also offer a parallel version of Balans, which is randomly generate several configurations of Balans, and run them parallely. 

```python
from balans.solver import ParBalans

instance_path = "tests/data/noswot.mps"
# how many Balans do you want to run
n_machines = 2

balans = ParBalans(instance_path=instance_path, 
                  n_machines=n_machines,
                  output_dir="results/") # where you want to save the output file
balans.solve()
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
* Random Objective[^6]
[^6]: Random Objective.
* Proximity Search[^7]
[^7]: M. Fischetti and M. Monaci. Proximity search for 0-1 mixed-integer convex programming. Journal of Heuristics, 20(6):709–731, Dec 2014.
* Crossover[^8]
[^8]: E. Rothberg. An Evolutionary Algorithm for Polishing Mixed Integer Programming Solutions. INFORMS Journal on Computing, 19(4):534–541, 2007.

## Available Repair Operators
* Repair MIP

## Installation

Balans requires Python 3.10+ can be installed from PyPI via `pip install balans`. 

## Test Your Setup

```
$ cd balans
$ python -m unittest discover tests
```

## Citation

If you use Balans in a publication, please cite it as:

```bibtex
    @inproceedings{balans25,
      author       = {Junyang Cai and
                      Serdar Kadioglu and
                      Bistra Dilkina},
      title        = {BALANS: Multi-Armed Bandits-based Adaptive Large Neighborhood Search for Mixed-Integer Programming Problems},
      booktitle    = {Proceedings of the Thirty-Fourth International Joint Conference on
                      Artificial Intelligence, {IJCAI} 2025, Montreal, Canada, August 16-22,
                      2025},
      pages        = {xx--xx},
      publisher    = {ijcai.org},
      year         = {2025},
      url          = {https://www.ijcai.org/proceedings/2025/xx},
    }
```

## License

Balans is licensed under the [Apache License 2.0](LICENSE).

<br>
