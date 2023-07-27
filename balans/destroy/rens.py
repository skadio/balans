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

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    lp_var_to_val = current.instance.lp_var_to_val

    float_index_to_be_bounded = [i for i in discrete_indexes if lp_var_to_val[i] % 1 != 0]

    print("\t Float set:", float_index_to_be_bounded)
    print("\t Destroy set:", next_state.destroy_set)

    return _State(next_state.instance,
                  next_state.var_to_val,
                  next_state.obj_val,
                  destroy_set=None,
                  float_index_to_be_bounded=float_index_to_be_bounded)
