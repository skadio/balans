import copy
from balans.base_state import _State
import math


def rens(current: _State, rnd_state) -> _State:
    #  Take an LP relaxed solution of the original MIP.
    #  For  discrete variables, choose the "non-discrete" ones
    #  and store them in the flaot index to be bounded list.
    #  Send the destroy set (None for the rens case) and
    #  float_index_to_be_bounded to base_instance.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    discrete_indexes = current.instance.discrete_indexes

    # Take an LP relaxed solution of the original MIP.
    lp_obj_val, lp_var_to_val = current.lp_obj_val, current.lp_var_to_val

    float_index_to_be_bounded = []
    for i in range(len(discrete_indexes)):
        if lp_var_to_val[i] % 1 != 0:
            float_index_to_be_bounded.append(i)

    next_state.destroy_set = None
    print("\t Float set:", float_index_to_be_bounded)
    print("\t Destroy set:", next_state.destroy_set)
    return _State(next_state.instance, next_state.var_to_val, next_state.obj_val,
                  next_state.destroy_set, float_index_to_be_bounded=float_index_to_be_bounded
                  , lp_obj_val=lp_obj_val, lp_var_to_val=lp_var_to_val)
