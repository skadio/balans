import copy
from balans.base_state import _State


def proximity(current: _State, rnd_state) -> _State:
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    discrete_indexes = current.instance.discrete_indexes
    binary_indexes = current.instance.binary_indexes

    # Take LP solution
    lp_var_to_val, lp_obj_val = current.instance.lp_solve()

    # for binary variable condition
    proximity_set = set(rnd_state.choice(binary_indexes, int(len(binary_indexes))))

    next_state.destroy_set = None

    print("\t Destroy set:", next_state.destroy_set)
    print("\t Binary set:", proximity_set)

    return _State(next_state.instance, next_state.var_to_val, next_state.obj_val, next_state.destroy_set,
                  proximity_set=proximity_set)
