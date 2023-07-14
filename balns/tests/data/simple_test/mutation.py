
import matplotlib.pyplot as plt
import numpy as np
from mabwiser.mab import LearningPolicy

from alns import ALNS
from alns.accept import *
from alns.select import *
from alns.stop import *
import pandas as pd
import math
import os
import pyscipopt as scip
from problemstate import ProblemState
from readinstance import ReadInstance

def extract_variable_features(state: ProblemState):
    varbls = state.model.getVars()
    var_types = [v.vtype() for v in varbls]
    lbs = [v.getLbGlobal() for v in varbls]
    ubs = [v.getUbGlobal() for v in varbls]

    type_mapping = {"BINARY": 0, "INTEGER": 1, "IMPLINT": 2, "CONTINUOUS": 3}
    var_types_numeric = [type_mapping.get(t, 0) for t in var_types]

    variable_features = pd.DataFrame({
        'var_type': var_types_numeric,
        'var_lb': lbs,
        'var_ub': ubs
    })

    variable_features = variable_features.astype({'var_type': int, 'var_lb': float, 'var_ub': float})

    return variable_features  # pd.dataframe


def to_destroy_mut(discrete) -> int:
    delta = 0.75
    return int(delta * len(discrete))


def find_discrete(state: ProblemState):
    discrete = []
    for i in range(state.length()):
        var_features = extract_variable_features(state)
        if var_features['var_type'][i] == 0 or var_features['var_type'][i] == 1:
            discrete.append(i)
    return discrete


def mutation_op(state: ProblemState, rnd_state):
    discrete = find_discrete(state)

    to_remove = rnd_state.choice(discrete, size=to_destroy_mut(discrete))

    # fix variable yapman lazim burada

    assignments = state.solution.copy()
    assignments[to_remove] = None
    # print(assignments)

    scip_sol = state.model.createSol()
    subMIP_vars = state.model.getVars()

    for i in range(state.model.getNVars()):
        val = assignments[i]
        state.model.setSolVal(scip_sol, subMIP_vars[i], val)

    #         for var in state.model.getVars()[var]:
    #             print("var",var)
    #             #state.model.setSolVal(scip_sol, var, assignments[var])

    return ProblemState(scip_sol, state.model)


def repair_op(state: ProblemState, rnd_state) -> ProblemState:
    # Not the most effective way but it works!
    # for i in to_remove:
    #   state.model.addCons(state.model.getVars()[i] == state.solution[i])

    #var.getIndex()
    for var in state.model.getVars():
        if not np.isnan(state.x[var]):
            # not the best way to enforce that, but it works
            state.model.addCons(var == state.x[var])
            # model.addCons(x  == 9)

            # model.addCons(x + y + z == 32, name="Heads")

    # s.getVal(x) == s.getSolVal(solution, x)

    #solve sub_MIP
    state.model.optimize()

    solution = state.model.getBestSol()

    state = ProblemState(solution, state.model)
    print(state.solution)
    return state
