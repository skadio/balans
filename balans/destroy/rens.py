import copy
from balans.base_state import _State
import math


def is_int(x):
    if x % 1 == 0:
        return True
    else:
        return False


def rens(current: _State, rnd_state) -> _State:
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    discrete_indexes = current.instance.discrete_indexes

    lp_var_to_val, lp_obj_val = current.instance.lp_solve()

    float_index_to_be_bounded = []
    for i in range(len(discrete_indexes)):
        if lp_var_to_val[i] % 1 != 0:
            float_index_to_be_bounded.append(i)

    next_state.destroy_set = None
    print("\t Float set:", float_index_to_be_bounded)
    print("\t Destroy set:", next_state.destroy_set)
    return _State(next_state.instance, next_state.var_to_val, next_state.obj_val, next_state.destroy_set,float_index_to_be_bounded=float_index_to_be_bounded)
