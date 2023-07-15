import matplotlib.pyplot as plt
import numpy as np
from mabwiser.mab import LearningPolicy

from alns import ALNS
from alns.accept import *
from alns.select import *
from alns.stop import *
import copy
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


def to_destroy_mut(discrete, delta) -> int:
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

    to_remove = rnd_state.choice(discrete, size=to_destroy_mut(discrete, delta=0.25))


    assignments = state.solution.copy()
    assignments[to_remove] = None

    subMIP_vars = state.model.getVars()

    for var in subMIP_vars:

        if var.getIndex() in to_remove:
            state.x[var] = 0

    return ProblemState(state.x, state.model)


def mutation_op2(state: ProblemState, rnd_state):
    #state = copy.deepcopy(state)

    discrete = find_discrete(state)

    to_remove = rnd_state.choice(discrete, size=to_destroy_mut(discrete, delta=0.50))


    assignments = state.solution.copy()
    assignments[to_remove] = None

    subMIP_vars = state.model.getVars()

    for var in subMIP_vars:

        if var.getIndex() in to_remove:
            state.x[var] = 0

    return ProblemState(state.x, state.model)


def mutation_op3(state: ProblemState, rnd_state):
    #state = copy.deepcopy(state)

    discrete = find_discrete(state)

    to_remove = rnd_state.choice(discrete, size=to_destroy_mut(discrete, delta=0.75))
    assignments = state.solution.copy()
    assignments[to_remove] = None
    sub_vars = state.model.getVars()
    for var in sub_vars:

        if var.getIndex() in to_remove:
            state.x[var] = 0

    return ProblemState(state.x, state.model)


def repair_op(state: ProblemState, rnd_state) -> ProblemState:

    for var in state.model.getVars():
        # if not np.isnan(state.x[var]):
        # print(state.x[var])
        if state.x[var] == 0:
            # not the best way to enforce that, but it works
            state.model.addCons(var == state.x[var])


    # solve sub_MIP
    state.model.optimize()

    solution = state.model.getBestSol()

    state = ProblemState(solution, state.model)
    print("current iteration: ", state.solution)
    print("current obj val: ", state.objective())

    return state
