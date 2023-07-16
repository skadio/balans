import copy
from balans.base_state import _State


def _mutation(current: _State, rnd_state, delta) -> _State:

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    discrete_indexes = current.instance.discrete_indexes
    destroy_size = int(delta * len(discrete_indexes))
    next_state.destroy_set = set(rnd_state.choice(discrete_indexes, destroy_size))

    print("\t Destroy set:", next_state.destroy_set)
    return next_state


def mutation_25(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.25)


def mutation_50(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.50)


def mutation_75(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.75)