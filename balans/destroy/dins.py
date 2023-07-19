import copy
from balans.base_state import _State


def dins(current: _State, rnd_state) -> _State:
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    discrete_indexes = current.instance.discrete_indexes
    binary_indexes = current.instance.binary_indexes


    # Take LP solution
    lp_var_to_val, lp_obj_val = current.instance.lp_solve()

    # Set J in algorithm has both binary and integer
    Set_J = []
    for i in range(len(discrete_indexes)):
        if abs(lp_var_to_val[i] - next_state.var_to_val[i]) >= 0.5:
            Set_J.append(i)

    #for binary variable condition
    dins_set = set(rnd_state.choice(binary_indexes, int(0.5 * len(binary_indexes))))

    next_state.destroy_set = set(Set_J)

    print("\t Destroy set:", next_state.destroy_set)
    print("\t Binary set:", dins_set)
    return _State(next_state.instance, next_state.var_to_val, next_state.obj_val, next_state.destroy_set,
                  dins_set=dins_set)
