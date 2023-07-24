import copy
from balans.base_state import _State

from balans.base_state import _State


# def crossover(state: _State, rnd_state):
#     sub_vars = state.model.getVars()
#     same_vars = []
#     for var in sub_vars:
#         if init_sol.x[var] == init_sol2.x[var]:
#             same_vars.append(var)
#
#     for var in same_vars:
#         state.x[var] = 0
#
#     return State(state.x, state.model)

def crossover(current: _State, rnd_state) -> _State:
    #  Take two random solutions.
    #  If a DISCRETE variable x_inc1 = x_inc2, do not change it.
    #  Otherwise, put it to the destroy set.
    #  Send the destroy set to base_instance.
    # TODO
    print("\t Destroy current objective:", current.obj_val)
    next_state = copy.deepcopy(current)

    discrete_indexes = current.instance.discrete_indexes

    destroy_size = int(len(discrete_indexes))

    next_state.destroy_set = set(rnd_state.choice(discrete_indexes, destroy_size))

    print("\t Destroy set:", next_state.destroy_set)
    return _State(next_state.instance, next_state.var_to_val, next_state.obj_val, next_state.destroy_set)
