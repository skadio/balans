import copy
from balans.base_state import _State


def dins(current: _State, rnd_state) -> _State:
    #  Take an LP relaxed solution of the original MIP.
    #  By considering only discrete variables, forms a Set J where |x_lp -x_inc| >=   0.5
    #  For binary variables we have a hard constraint,
    #  here we say change at most half of them.
    #  If a variable is inside the Set J, it is part of the destroy set.
    #  Send the destroy set (Set J) and dins_set to base_instance.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    discrete_indexes = current.instance.discrete_indexes
    binary_indexes = current.instance.binary_indexes

    # Take an LP relaxed solution of the original MIP.
    lp_obj_val, lp_var_to_val = current.lp_obj_val, current.lp_var_to_val

    # By considering only discrete variables, forms a Set J where |x_lp -x_inc| >=   0.5
    Set_J = []
    for i in range(len(discrete_indexes)):
        if abs(lp_var_to_val[i] - next_state.var_to_val[i]) >= 0.5:
            Set_J.append(i)

    # For binary variables we have a hard constraint, here we say change at most half of them.
    dins_set = set(rnd_state.choice(binary_indexes, int(0.5 * len(binary_indexes))))

    # If a variable is inside the Set J, it is part of the destroy set.
    next_state.destroy_set = set(Set_J)

    print("\t Destroy set:", next_state.destroy_set)
    print("\t Binary set:", dins_set)
    return _State(next_state.instance, next_state.var_to_val, next_state.obj_val, next_state.destroy_set,
                  dins_set=dins_set,lp_obj_val=lp_obj_val, lp_var_to_val=lp_var_to_val)


