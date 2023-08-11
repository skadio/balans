import copy

import numpy as np

from balans.base_state import _State
from balans.utils_scip import random_solve


# 3 DIFFERENT VERSIONS OF CROSS OVER IMPLEMENTED, ORIGINAL ONE IS CROSSOVER.

def crossover(current: _State, rnd_state) -> _State:
    #  Take two random solutions.
    #  If a DISCRETE variable i, x_rand[i] = x_rand2[i], do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes

    # Dynamic 2 RANDOM SOLUTIONS
    a = rnd_state.uniform(low=0.3, high=0.95, size=1)
    b = rnd_state.uniform(low=0.40, high=0.90, size=1)
    random_index_to_val, random_obj_value = random_solve(path=current.instance.path, gap=a[0])
    random2_index_to_val, random2_obj_value = random_solve(path=current.instance.path, gap=b[0])

    #  If a variable x_rand = x_rand2, do not change it.
    indexes_with_same_value = [i for i in discrete_indexes if
                               np.isclose(random_index_to_val[i], random2_index_to_val[i])]

    # Else potentially change it
    crossover_set = set([i for i in discrete_indexes if i not in indexes_with_same_value])

    # Update the next index
    next_state.index_to_val = random_index_to_val

    print("\t Destroy set:", crossover_set)
    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=crossover_set)


def crossover1(current: _State, rnd_state) -> _State:
    #  Take two random*** (CURRENTLY LAST TWO SOLUTION) solutions.
    #  If a DISCRETE variable i, x_inc[i] = x_inc2[i], do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    random_sol = current.instance.random_index_to_val

    #  If a variable x_inc = x_inc2, do not change it.
    indexes_with_same_value = [i for i in discrete_indexes if
                               np.isclose(current.previous_index_to_val[i], next_state.index_to_val[i])]

    # Else potentially change it
    crossover_set = set([i for i in discrete_indexes if i not in indexes_with_same_value])
    print("\t Selected Operator: ", "crossover")
    print("\t Destroy set:", crossover_set)

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=crossover_set,
                  previous_index_to_val=current.index_to_val)


def crossover2(current: _State, rnd_state) -> _State:
    #  Take one random and current incumbent solutions.
    #  If a DISCRETE variable i, x_inc[i] = x_rand[i], do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    random_sol = current.instance.random_index_to_val

    #  If a variable x_inc = x_rand, do not change it.
    indexes_with_same_value = [i for i in discrete_indexes if
                               np.isclose(random_sol[i], next_state.index_to_val[i])]

    # Else potentially change it
    crossover_set = set([i for i in discrete_indexes if i not in indexes_with_same_value])

    print("\t Destroy set:", crossover_set)

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=crossover_set)
