import copy
from base_state import State


def _mutation(current: State, rnd_state, delta) -> State:

    next_state = copy.deepcopy(current)

    discrete_indexes = current.instance.discrete_indexes
    destroy_size = int(delta * len(discrete_indexes))
    next_state.destroy_set = set(rnd_state.choice(discrete_indexes, destroy_size))

    return next_state


def mutation_25(current: State, rnd_state) -> State:
    return _mutation(current, rnd_state, delta=0.25)


def mutation_50(current: State, rnd_state) -> State:
    return _mutation(current, rnd_state, delta=0.50)


def mutation_75(current: State, rnd_state) -> State:
    return _mutation(current, rnd_state, delta=0.75)