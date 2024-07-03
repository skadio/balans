import copy

import math
import pyscipopt as scip

from balans.utils_scip import get_index_to_val_and_objective, sort_list_with_indices
from balans.base_state import _State


# 3 DIFFERENT VERSIONS OF CROSS OVER IMPLEMENTED, ORIGINAL ONE IS CROSSOVER.
def crossover(current: _State, rnd_state) -> _State:
    #  Take one RANDOM solutions.
    #  If a DISCRETE variable x_rand = x_inc, do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.
    print("*** Operator: ", "CROSSOVER")
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes

    # Dynamic Random Solution
    org_objective = current.instance.model.getObjective()
    variables = current.instance.model.getVars()
    objective = scip.Expr()
    for var in variables:
        coeff = rnd_state.uniform(-1, 1)
        if coeff != 0:
            objective += coeff * var
    objective.normalize()
    current.instance.model.setObjective(objective, current.instance.sense)
    current.instance.model.setParam("limits/bestsol", 1)
    current.instance.model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

    # Solve
    current.instance.model.optimize()

    if current.instance.model.getNSols() == 0:
        current.instance.model.freeTransform()
        current.instance.model.setParam("limits/bestsol", -1)
        current.instance.model.setObjective(org_objective, current.instance.sense)
        current.instance.model.setHeuristics(scip.SCIP_PARAMSETTING.DEFAULT)
        return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=[])

    r1_index_to_val, r1_obj_val = get_index_to_val_and_objective(current.instance.model)

    # Get back the original model
    current.instance.model.freeTransform()
    current.instance.model.setParam("limits/bestsol", -1)
    current.instance.model.setObjective(org_objective, current.instance.sense)
    current.instance.model.setHeuristics(scip.SCIP_PARAMSETTING.DEFAULT)
    # print("Random Solution1:", r1_index_to_val)

    #  If a discrete variable x_rand1 = x_inc, do not change it.
    indexes_with_same_value = [i for i in discrete_indexes if
                               math.isclose(r1_index_to_val[i], current.index_to_val[i])]

    # Else potentially change it
    crossover_set = set([i for i in discrete_indexes if i not in indexes_with_same_value])

    next_state.destroy_set = crossover_set

    # print("\t Destroy set:", crossover_set)
    return next_state


def crossover_random(current: _State, rnd_state, delta) -> _State:
    #  Take one RANDOM solutions.
    #  If a DISCRETE variable x_rand = x_inc, do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.
    print("*** Operator: ", "CROSSOVER")
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes

    # Dynamic Random Solution
    org_objective = current.instance.model.getObjective()
    variables = current.instance.model.getVars()
    objective = scip.Expr()
    for var in variables:
        coeff = rnd_state.uniform(-1, 1)
        if coeff != 0:
            objective += coeff * var
    objective.normalize()
    current.instance.model.setObjective(objective, current.instance.sense)
    current.instance.model.setParam("limits/bestsol", 1)
    current.instance.model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

    # Solve
    current.instance.model.optimize()

    if current.instance.model.getNSols() == 0:
        current.instance.model.freeTransform()
        current.instance.model.setParam("limits/bestsol", -1)
        current.instance.model.setObjective(org_objective, current.instance.sense)
        current.instance.model.setHeuristics(scip.SCIP_PARAMSETTING.DEFAULT)
        return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=[])

    r1_index_to_val, r1_obj_val = get_index_to_val_and_objective(current.instance.model)

    # Get back the original model
    current.instance.model.freeTransform()
    current.instance.model.setParam("limits/bestsol", -1)
    current.instance.model.setObjective(org_objective, current.instance.sense)
    current.instance.model.setHeuristics(scip.SCIP_PARAMSETTING.DEFAULT)
    # print("Random Solution1:", r1_index_to_val)

    #  If a discrete variable x_rand1 = x_inc, do not change it.
    indexes_with_same_value = [i for i in discrete_indexes if
                               math.isclose(r1_index_to_val[i], current.index_to_val[i])]

    # Else potentially change it
    non_zero_set = set([i for i in discrete_indexes if i not in indexes_with_same_value])
    zero_set = set([i for i in discrete_indexes if i in indexes_with_same_value])
    print(non_zero_set)

    size = min(int(delta * current.adaptive * len(discrete_indexes)),
               int(current.max_fraction * len(discrete_indexes)))
    if len(non_zero_set) >= size:
        diff = [abs(r1_index_to_val[i] - current.index_to_val[i]) if i in discrete_indexes else 0
                for i in range(len(r1_index_to_val))]
        diff_value, diff_index = sort_list_with_indices(diff)
        crossover_set = set(diff_index[:size])
    else:
        random_set = set(rnd_state.choice(list(zero_set), size - len(non_zero_set), replace=False))
        crossover_set = non_zero_set | random_set

    next_state.destroy_set = crossover_set
    print(crossover_set)
    # print("\t Destroy set:", crossover_set)
    return next_state


def crossover_random_25(current: _State, rnd_state) -> _State:
    return crossover_random(current, rnd_state, delta=0.25)

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



