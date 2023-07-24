import copy
from balans.base_state import _State


def no_objective(current: _State, rnd_state) -> _State:

    # CHANGE OBJECTIVE TO ZERO OBJECTIVE
    # is_zero_obj sends non-empty list to base instance to do that.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    discrete_indexes = current.instance.discrete_indexes

    # Will be used later to improve its performance through delta*obj_lp_relax constraint
    lp_var_to_val, lp_obj_val = current.instance.lp_solve()

    next_state.destroy_set = None

    # Just to make "is_zero_obj" not None
    is_zero_obj = [1]

    print("\t Destroy set:", next_state.destroy_set)
    return _State(next_state.instance, next_state.var_to_val, next_state.obj_val, next_state.destroy_set,
                  is_zero_obj=is_zero_obj)
