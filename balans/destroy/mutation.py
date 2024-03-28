import copy

from balans.base_state import _State


def _mutation(current: _State, rnd_state, delta) -> _State:
    # Take the discrete variables of the original problem
    # Then, depending on delta parameter, choose a random subset of discrete variables to destroy.
    # Send the destroy set to base_instance.

    print("\t Selected Operator: ", "mutation")
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    discrete_indexes = current.instance.discrete_indexes
    destroy_size = int(delta * len(discrete_indexes))

    print("\t Discrete index:", discrete_indexes)

    mutation_set = set(rnd_state.choice(discrete_indexes, destroy_size))

    print("\t Destroy set:", mutation_set)

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=mutation_set)


def mutation_25(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.25)


def mutation_50(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.50)


def mutation_75(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.75)


def _mutation_binary(current: _State, rnd_state, delta) -> _State:
    # Take the BINARY variables of the original problem
    # Then, depending on delta parameter, choose a random subset of discrete variables to destroy.
    # Send the destroy set to base_instance.

    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)
    next_state.reset_solve_settings()

    # Static features from the instance
    binary_indexes = current.instance.binary_indexes

    destroy_size = int(delta * len(binary_indexes))

    # Select a subset of binary variables end fix other binary variables.
    mutation_binary_set = set(rnd_state.choice(binary_indexes, destroy_size))

    print("\t Destroy set:", mutation_binary_set)

    return _State(next_state.instance,
                  next_state.index_to_val,
                  next_state.obj_val,
                  destroy_set=mutation_binary_set)


def mutation_binary_25(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.25)


def mutation_binary_50(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.50)


def mutation_binary_75(current: _State, rnd_state) -> _State:
    return _mutation(current, rnd_state, delta=0.75)

