import copy
from balans.base_state import _State


def dins_randomized(current: _State, rnd_state, delta) -> _State:
    #  Take an LP relaxed solution of the original MIP.
    #  By considering only discrete variables, forms a Set J where |x_lp -x_inc| >=   0.5
    #  For binary variables we have a hard constraint,
    #  here we say change at most half of them.
    #  If a variable is inside the Set J, it is part of the destroy set.
    #  Send the destroy set (Set J) and dins_set to base_instance.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    binary_indexes = current.instance.binary_indexes
    lp_index_to_val = current.instance.lp_index_to_val

    print("\t discrete_indexes:", discrete_indexes)
    print("\t lp_index_to_val: ", lp_index_to_val)

    # By considering only discrete variables, form a set_j where |x_lp - x_inc| >= 0.5
    set_j = set([i for i in discrete_indexes
                 if abs(lp_index_to_val[i] - current.index_to_val[i]) >= 0.5])

    # Randomization STEP for set_j
    fix_size = int(delta * len(set_j))
    random_set_j = set(rnd_state.choice(set_j, fix_size))

    # DINS for binary: hard constraint to change at most half of the binary variables
    dins_binary_set = set(rnd_state.choice(binary_indexes, int(delta * len(binary_indexes))))

    print("\t Destroy set:", random_set_j)
    print("\t DINS Binary set:", dins_binary_set)

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=random_set_j,
                  dins_binary_set=dins_binary_set)


def dins(current: _State, rnd_state, delta) -> _State:
    #  Take an LP relaxed solution of the original MIP.
    #  By considering only discrete variables, forms a Set J where |x_lp -x_inc| >=   0.5
    #  For binary variables we have a hard constraint,
    #  here we say change at most half of them.
    #  If a variable is inside the Set J, it is part of the destroy set.
    #  Send the destroy set (Set J) and dins_set to base_instance.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    binary_indexes = current.instance.binary_indexes
    lp_index_to_val = current.instance.lp_index_to_val

    print("\t discrete_indexes:", discrete_indexes)
    print("\t lp_index_to_val: ", lp_index_to_val)

    # By considering only discrete variables, form a set_j where |x_lp - x_inc| >= 0.5
    set_j = set([i for i in discrete_indexes
                 if abs(lp_index_to_val[i] - current.index_to_val[i]) >= 0.5])

    # DINS for binary: hard constraint to change at most half of the binary variables
    dins_binary_set = set(rnd_state.choice(binary_indexes, int(delta * len(binary_indexes))))

    print("\t Destroy set:", set_j)
    print("\t DINS Binary set:", dins_binary_set)

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=set_j,
                  dins_binary_set=dins_binary_set)


def dins_50(current: _State, rnd_state) -> _State:
    return dins(current, rnd_state, delta=0.50)


def dins_75(current: _State, rnd_state) -> _State:
    return dins(current, rnd_state, delta=0.75)
