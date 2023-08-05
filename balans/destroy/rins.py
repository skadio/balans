import copy
from balans.base_state import _State
import numpy as np


def rins(current: _State, rnd_state) -> _State:
    #  Take the LP relaxed solution of the original MIP and the current incumbent.
    #  If a DISCRETE variable x_inc = x_lp, then do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    lp_index_to_val = current.instance.lp_index_to_val

    #  If a variable x_inc = x_lp, do not change it.
    indexes_with_same_value = [i for i in discrete_indexes if
                               np.isclose(lp_index_to_val[i], next_state.index_to_val[i])]

    # Else potentially change it
    destroy_set = set([i for i in discrete_indexes if i not in indexes_with_same_value])

    print("\t Destroy set:", destroy_set)
    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=destroy_set)


def _rins_random(current: _State, rnd_state, delta) -> _State:
    #  Take the LP relaxed solution of the original MIP and the current incumbent.
    #  If a DISCRETE variable x_inc = x_lp, then do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    lp_index_to_val = current.instance.lp_index_to_val

    #  If a variable x_inc = x_lp, do not change it.
    indexes_with_same_value = [i for i in discrete_indexes if
                               np.isclose(lp_index_to_val[i], next_state.index_to_val[i])]

    # Randomization STEP
    fix_size = int(delta * len(indexes_with_same_value))
    fix_set = set(rnd_state.choice(indexes_with_same_value, fix_size))

    # Else potentially change it
    destroy_set = set([i for i in discrete_indexes if i not in fix_set])

    print("\t Destroy set:", destroy_set)
    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=destroy_set)


def rins_random_50(current: _State, rnd_state) -> _State:
    return _rins_random(current, rnd_state, delta=0.50)
