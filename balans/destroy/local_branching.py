import copy
from balans.base_state import _State


def _local_branching(current: _State, rnd_state, delta) -> _State:
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    binary_indexes = current.instance.binary_indexes
    destroy_size = int(delta * len(binary_indexes))

    next_state.destroy_set = set(rnd_state.choice(binary_indexes, destroy_size))

    print("\t Destroy set:", next_state.destroy_set)
    return _State(next_state.instance, next_state.var_to_val, next_state.obj_val, next_state.destroy_set)


def local_branching_25(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.25)


def local_branching_50(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.50)


def local_branching_75(current: _State, rnd_state) -> _State:
    return _local_branching(current, rnd_state, delta=0.75)
