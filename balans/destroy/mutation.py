import copy
from balans.base_state import _State


def _mutation(current: _State, rnd_state, delta) -> _State:
    #  Take the discrete variables of the original problem
    # Then, depending on delta parameter, choose a random subset of discrete variables to destroy.
    #  Send the destroy set to base_instance.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    print("\t Discrete:", discrete_indexes)

    destroy_size = int(delta * len(discrete_indexes))

    mutated_set = set(rnd_state.choice(discrete_indexes, destroy_size))

    print("\t Destroy set:", mutated_set)

    return _State(next_state.instance,
                  next_state.var_to_val,
                  next_state.obj_val,
                  destroy_set=mutated_set)


def mutation_25(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.25)


def mutation_50(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.50)


def mutation_75(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.75)


# TODO problematic when destroy size = num discrete
def mutation_100(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=1.0)
