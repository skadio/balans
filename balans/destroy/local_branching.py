import copy

from balans.base_state import _State


def _local_branching(current: _State, rnd_state, delta) -> _State:
    #  For binary variables we have a hard constraint,
    #  here we say change at most half of them (delta=0.5).
    #  Other possible delta values are 0.25 and 0.75.
    #  Send the destroy set to base_instance.

    print("*** Operator: ", "LOCAL BRANCHING")
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    binary_indexes = current.instance.binary_indexes

    # <= k in local branching
    next_state.local_branching_size = rnd_state.randint(int(0.05 * len(binary_indexes)),
                                                        int(delta * len(binary_indexes)))

    return next_state


def local_branching_10(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.10)


def local_branching_25(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.25)


def local_branching_50(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.50)
