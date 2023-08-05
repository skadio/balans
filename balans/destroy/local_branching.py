import copy
from balans.base_state import _State


def _local_branching(current: _State, rnd_state, delta) -> _State:
    #  For binary variables we have a hard constraint,
    #  here we say change at most half of them (delta=0.5).
    #  Other possible delta values are 0.25 and 0.75.
    #  Send the destroy set to base_instance.
    #  Note: These indexes are determined by the solver in this implementation.
    # Please see the base_instance is_local_branching part. Operations are implemented inside that.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    binary_indexes = current.instance.binary_indexes

    # <= k in local branching
    local_branching_size = int(delta * len(binary_indexes))

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  local_branching_size=local_branching_size)


def local_branching_50(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.50)
