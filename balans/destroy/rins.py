import copy
from balans.base_state import _State


def rins(current: _State, rnd_state) -> _State:
    #  Take an LP relaxed solution of the original MIP and current incumbent.
    #  If a DISCRETE variable x_inc = x_lp, do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    discrete_indexes = current.instance.discrete_indexes

    # Take an LP relaxed solution of the original MIP.
    lp_obj_val, lp_var_to_val = current.lp_obj_val, current.lp_var_to_val

    #  If a variable x_inc = x_lp, do not change it.
    same_index = []
    for i in range(len(discrete_indexes)):
        if lp_var_to_val[i] == next_state.var_to_val[i]:
            same_index.append(i)

    #  Otherwise, put it to the destroy set.
    destroy_set = []
    for i in discrete_indexes:
        if i not in same_index:
            destroy_set.append(i)

    next_state.destroy_set = set(destroy_set)

    print("\t Destroy set:", next_state.destroy_set)
    return _State(next_state.instance, next_state.var_to_val, next_state.obj_val,
                  next_state.destroy_set,lp_obj_val=lp_obj_val, lp_var_to_val=lp_var_to_val)

