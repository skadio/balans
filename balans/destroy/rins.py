import copy
from balans.base_state import _State


def rins(current: _State, rnd_state) -> _State:
    #  Take an LP relaxed solution of the original MIP and current incumbent.
    #  If a DISCRETE variable x_inc = x_lp, do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    lp_var_to_val = current.instance.lp_var_to_val

    #  If a variable x_inc = x_lp, do not change it.
    same_index = [i for i in discrete_indexes if lp_var_to_val[i] == next_state.var_to_val[i]]

    # Else potentially change it.
    destroy_set = set([i for i in discrete_indexes if i not in same_index])

    print("\t Destroy set:", destroy_set)
    return _State(next_state.instance,
                  next_state.var_to_val,
                  next_state.obj_val,
                  destroy_set=destroy_set)
