import copy
import math

from balans.base_state import _State


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

    r1_index_to_val, _ = current.instance.mip.solve_random_and_undo()

    # If we don't find a random feasible solution
    if len(r1_index_to_val) == 0:
        return next_state

    #  If a discrete variable x_rand1 = x_inc, do not change it.
    indexes_with_same_value = [i for i in discrete_indexes if
                               math.isclose(r1_index_to_val[i], current.index_to_val[i])]

    # Else potentially change it
    next_state.destroy_set = set([i for i in discrete_indexes if i not in indexes_with_same_value])

    return next_state



