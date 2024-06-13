import copy

import math

from balans.base_state import _State
from balans.utils_scip import random_solve


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

    # Dynamic Random Solution
    r1_seed = rnd_state.tomaxint()
    r1_index_to_val, r1_obj_val = random_solve(path=current.instance.path, scip_seed=r1_seed,
                                               has_random_obj=True, solution_count=1)

    print("Random Solution1:", r1_index_to_val)

    #  If a discrete variable x_rand1 = x_inc, do not change it.
    indexes_with_same_value = [i for i in discrete_indexes if
                               math.isclose(r1_index_to_val[i], current.index_to_val[i])]

    # Else potentially change it
    crossover_set = set([i for i in discrete_indexes if i not in indexes_with_same_value])

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
#     print("\t Destroy current objective:", current.obj_value)
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
#                   next_state.obj_value,
#                   destroy_set=crossover_set)

# def crossover3(current: _State, rnd_state) -> _State:
#     #  Take two random*** (CURRENTLY LAST TWO SOLUTION) solutions.
#     #  If a DISCRETE variable x_inc = x_inc2, do not change it.
#     #  Otherwise, put it to the destroy set.
#     #  Send the destroy set to base_instance.
#     print("\t Destroy current objective:", current.obj_value)
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
#                   next_state.obj_value,
#                   destroy_set=crossover_set,
#                   previous_index_to_val=current.index_to_val)
#



