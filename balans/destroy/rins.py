import copy
import math

from balans.base_state import _State


def _rins(current: _State, rnd_state, delta) -> _State:
    #  Take the LP relaxed solution of the original MIP and the current incumbent.
    #  If a DISCRETE variable x_inc = x_lp, then do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.

    print("*** Operator: ", "RINS")
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    lp_index_to_val = current.instance.lp_index_to_val

    #  If a variable x_cur = x_lp, do not change it.
    indexes_with_diff_value = [i for i in discrete_indexes if not
                               math.isclose(lp_index_to_val[i], current.index_to_val[i])]

    # Else potentially change it
    size = int(delta * len(indexes_with_diff_value))
    next_state.destroy_set = set(rnd_state.choice(indexes_with_diff_value, size))

    return next_state


def rins_50(current: _State, rnd_state) -> _State:
    return _rins(current, rnd_state, delta=0.50)
