import copy
from balans.base_state import _State


def _local_branching(current: _State, rnd_state, delta) -> _State:
    #  For discrete variables we have a hard constraint,
    #  here we say change at most half of them (delta=0.5).
    #  Other possible delta values are 0.25 and 0.75.
    #  Send the destroy set to base_instance.
    # TODO for FUTURE NOTE:
    #  Here we ourselves give (FORCE) free indexes to the solver
    # TODO These indexes can also be determined by the solver.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    binary_indexes = current.instance.binary_indexes
    destroy_size = int(delta * len(binary_indexes))

    # Select a subset of binary variables end fix other binary variables.
    local_branching_destroy_set = set(rnd_state.choice(binary_indexes, destroy_size))

    print("\t Destroy set:", local_branching_destroy_set)

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=local_branching_destroy_set)


def local_branching_v2(current: _State, rnd_state) -> _State:
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  is_local_branching=True)


def local_branching_25(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.25)


def local_branching_50(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.50)


def local_branching_75(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.75)
