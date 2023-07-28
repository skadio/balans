import copy
from balans.base_state import _State


def proximity(current: _State, rnd_state) -> _State:
    # TODO THE HEURISTIC DO NOT PROPERLY ITERATE YET.
    # For discrete variables we have a hard constraint,
    # Change the objective coefficients of the original
    # problem based on the current solution value.
    # For binary variables,
    # if x_inc =0, update its objective coefficient to 1.
    # if x_inc =1, update its objective coefficient to -1.
    # Drop all other variables by making their coefficient 0.
    # Send the destroy set to base_instance.
    # Note : the required objective operations for proximity search happens in base_instance file.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    binary_indexes = current.instance.binary_indexes
    lp_index_to_val = current.instance.lp_index_to_val

    # for binary variable condition
    proximity_set = set(rnd_state.choice(binary_indexes, int(len(binary_indexes))))

    proximity_destroy_set = None

    print("\t Destroy set:", proximity_destroy_set)
    print("\t Binary set:", proximity_set)

    return _State(next_state.instance, next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=proximity_destroy_set,
                  proximity_set=proximity_set)
