import copy

import numpy as np

from balans.base_state import _State
from balans.utils_scip import random_solve
from balans.utils import Constants


# 3 DIFFERENT VERSIONS OF CROSS OVER IMPLEMENTED, ORIGINAL ONE IS CROSSOVER.
def crossover(current: _State, rnd_state) -> _State:
    #  Take TWO RANDOM solutions.
    #  If a DISCRETE variable x_rand = x_rand2, do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.
    print("*** Operator: ", "CROSSOVER")
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes

    # Dynamic 2 RANDOM SOLUTIONS
    random_seed1 = rnd_state.tomaxint()
    random_gap1 = rnd_state.uniform(low=0.1, high=Constants.random_gap_ub1)

    random_seed2 = rnd_state.tomaxint()
    random_gap2 = rnd_state.uniform(low=0.1, high=Constants.random_gap_ub2)

    random1_index_to_val, random1_obj_value = random_solve(path=current.instance.path, scip_seed=random_seed1, gap=random_gap1)
    random2_index_to_val, random2_obj_value = random_solve(path=current.instance.path, scip_seed=random_seed2, gap=random_gap2)

    print("Random Solution1:", random1_index_to_val)
    print("Random Solution2:", random2_index_to_val)

    #  If a variable x_rand1 = x_rand2, do not change it.
    indexes_with_same_value = [i for i in discrete_indexes if
                               np.isclose(random1_index_to_val[i], random2_index_to_val[i])]

    # Else potentially change it
    crossover_set = set([i for i in discrete_indexes if i not in indexes_with_same_value])

    # Update the next index
    next_state.index_to_val = random1_index_to_val
    next_state.obj_val = random1_obj_value

    print("\tIndex to val:", next_state.index_to_val)

    print("\t Destroy set:", crossover_set)
    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=crossover_set)


# def crossover2(current: _State, rnd_state) -> _State:
#     #  Take one random and current incumbent solutions.
#     #  If a DISCRETE variable x_inc = x_rand, do not change it.
#     #  Otherwise, put it to the destroy set.
#     #  Send the destroy set to base_instance.
#     print("\t Destroy current objective:", current.obj_val)
#     next_state = copy.deepcopy(current)
#     next_state.reset_solve_settings()
#
#     # Static features from the instance
#     discrete_indexes = current.instance.discrete_indexes
#     random_sol = current.instance.random_index_to_val
#
#     print("Random Solution Crossover:", random_sol)
#
#     #  If a variable x_inc = x_rand, do not change it.
#     indexes_with_same_value = [i for i in discrete_indexes if
#                                math.isclose(random_sol[i], next_state.index_to_val[i])]
#
#     # Else potentially change it
#     crossover_set = set([i for i in discrete_indexes if i not in indexes_with_same_value])
#     print("\tIndex to val:", next_state.index_to_val)
#
#     print("\t Destroy set:", crossover_set)
#     return _State(next_state.instance,
#                   next_state.index_to_val,
#                   next_state.obj_val,
#                   destroy_set=crossover_set)

# def crossover3(current: _State, rnd_state) -> _State:
#     #  Take two random*** (CURRENTLY LAST TWO SOLUTION) solutions.
#     #  If a DISCRETE variable x_inc = x_inc2, do not change it.
#     #  Otherwise, put it to the destroy set.
#     #  Send the destroy set to base_instance.
#     print("\t Destroy current objective:", current.obj_val)
#     next_state = copy.deepcopy(current)
#     next_state.reset_solve_settings()
#
#     # Static features from the instance
#     discrete_indexes = current.instance.discrete_indexes
#     random_sol = current.instance.random_index_to_val
#
#     print("Random Solution Crossover:", random_sol)
#     #  If a variable x_inc = x_inc2, do not change it.
#     # TODO This probably has a BUG. Where was previous_index updated before coming here?
#     # Previous solution is only initialized in the root note in State creation but then never updated
#     indexes_with_same_value = [i for i in discrete_indexes if
#                                math.isclose(current.previous_index_to_val[i], next_state.index_to_val[i])]
#
#     # Else potentially change it
#     crossover_set = set([i for i in discrete_indexes if i not in indexes_with_same_value])
#     print("\t Previous Index to val:", next_state.index_to_val)
#
#     print("\t Destroy set:", crossover_set)
#     return _State(next_state.instance,
#                   next_state.index_to_val,
#                   next_state.obj_val,
#                   destroy_set=crossover_set,
#                   previous_index_to_val=current.index_to_val)
#



