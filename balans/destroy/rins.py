import numpy as np
from balans.base_state import _State


def to_destroy_rins(init_sol, init_sol2, discrete):
    to_remove = np.where(np.in1d(init_sol, init_sol2))[0]
    return to_remove


def lp_relax(state: _State):
    """
    Gets and Solves LP relaxed version of the same problem

    Returns
    -------
    objective value=float
    solution =array
    len_sol=int
    """
    vars = state.model.getVars()
    for v in vars:
        # Continuous relaxation of the problem
        state.model.chgVarType(v, 'CONTINUOUS')
    state.model.optimize()

    solution = state.model.getBestSol()

    state = _State(solution, state.model)
    print("current iteration: ", state.solution)
    print("current obj val: ", state.objective())

    return state


def rins(state: _State, rnd_state):
    discrete = find_discrete(state)
    lp_state = lp_relax(state)
    to_remove = rnd_state.choice(discrete, size=to_destroy_rins(discrete))

    assignments = state.solution.copy()
    assignments[to_remove] = None
    # print(assignments)

    subMIP_vars = state.model.getVars()
    same_vars = []
    for var in subMIP_vars:
        if lp_state.x[var] == state.x[var]:
            same_vars.append(var)

    for var in same_vars:
        state.x[var] = 0

