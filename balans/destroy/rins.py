import copy
from balans.base_state import _State


def rins(current: _State, rnd_state) -> _State:
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    discrete_indexes = current.instance.discrete_indexes

    lp_var_to_val, lp_obj_val = current.instance.lp_solve()

    same_index = []
    for i in range(len(discrete_indexes)):
        if lp_var_to_val[i] == next_state.var_to_val[i]:
            same_index.append(i)

    destroy_set = []
    for i in discrete_indexes:
        if i not in same_index:
            destroy_set.append(i)

    next_state.destroy_set = set(destroy_set)

    print("\t Destroy set:", next_state.destroy_set)
    return _State(next_state.instance, next_state.var_to_val, next_state.obj_val, next_state.destroy_set)

