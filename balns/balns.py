import numpy as np
from mabwiser.mab import LearningPolicy
from alns import ALNS
from alns.accept import *
from alns.select import *
from alns.stop import *
from problemstate import ProblemState
from readinstance import ReadInstance
from mutation import mutation_op, mutation_op2, mutation_op3, to_destroy_mut, find_discrete, extract_variable_features, \
    repair_op
from cross import crossover_op
from rins import rins_op


SEED = 42
np.random.seed(SEED)

if __name__ == "__main__":
    instance_path = "neos-5140963-mincio.mps.gz"

    # Terrible - but simple - two first solution, where only the first item is
    # selected.
    instance = ReadInstance(problem_instance_file=instance_path)
    instance2 = ReadInstance(problem_instance_file=instance_path)

    # Time =30 and gap limit = 50 percent gap within the solution
    init_sol = instance.initial_state(0.50, 30)
    # Time =30 and gap limit = 75 percent gap within the solution
    init_sol2 = instance2.initial_state(0.75, 30)

    print("Initial Feasible Solution:", init_sol.transform_solution_to_array2())
    print("Initial Objective Value:", init_sol.objective())

    print("Initial Feasible Solution:", init_sol2.transform_solution_to_array2())
    print("Initial Objective Value:", init_sol2.objective())


def make_alns() -> ALNS:
    rnd_state = np.random.RandomState(SEED)
    alns = ALNS(rnd_state)

    # noinspection PyTypeChecker
    alns.add_destroy_operator(mutation_op)
    # noinspection PyTypeChecker
    alns.add_destroy_operator(mutation_op2)
    # noinspection PyTypeChecker
    alns.add_destroy_operator(mutation_op3)
    # noinspection PyTypeChecker
    alns.add_destroy_operator(crossover_op)
    # noinspection PyTypeChecker
    alns.add_destroy_operator(rins_op)
    # noinspection PyTypeChecker
    alns.add_repair_operator(repair_op)
    return alns


accept = HillClimbing()

# MABSelector
# noinspection PyTypeChecker
select = MABSelector(scores=[5, 2, 1, 0.5],
                     num_destroy=5,
                     num_repair=1,
                     learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))

alns = make_alns()

res = alns.iterate(init_sol, select, accept, MaxIterations(5))

print(f"Found solution with objective {res.best_state.objective()}.")
