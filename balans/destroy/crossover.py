import copy
from balans.base_state import _State

from balans.base_state import _State


def crossover(current: _State, rnd_state) -> _State:
    # TODO THE HEURISTIC DO NOT PROPERLY ITERATE YET.
    #  Take two random solutions.
    #  If a DISCRETE variable x_inc1 = x_inc2, do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes

    destroy_size = int(len(discrete_indexes))

    crossover_destroy_set = set(rnd_state.choice(discrete_indexes, destroy_size))

    print("\t Destroy set:", next_state.destroy_set)
    return _State(next_state.instance,
                  next_state.var_to_val,
                  next_state.obj_val,
                  destroy_set=crossover_destroy_set)
