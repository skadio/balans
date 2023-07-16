import copy
from balans.base_state import _State


def rens(state: _State, rnd_state):
    sub_vars = state.model.getVars()
    same_vars = []
    for var in sub_vars:
        if init_sol.x[var] == init_sol2.x[var]:
            same_vars.append(var)

    for var in same_vars:
        state.x[var] = 0

    return _State(state.x, state.model)
