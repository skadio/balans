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
# from balns import solve

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


def find_discrete_index(state: ProblemState):
    discrete = []
    for i in range(state.length()):
        var_features = extract_variable_features(state)
        if var_features['var_type'][i] == 0 or var_features['var_type'][i] == 1:
            discrete.append(i)
    return discrete


def mutation_op(current: ProblemState, rnd_state):

    destroy = copy.deepcopy(current)

    destroy_index = find_discrete_index(destroy)

    # 5 , 7 , 9
    # [0, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    destroy.destroy_set = set(rnd_state.choice(destroy_index, size=to_destroy_mut(destroy_index, delta=0.25)))

    return destroy


def mutation_op2(current: ProblemState, rnd_state):
    #state = copy.deepcopy(state)

    discrete = find_discrete_index(current)

    to_remove = rnd_state.choice(discrete, size=to_destroy_mut(discrete, delta=0.50))

    assignments = current.solution.copy()
    assignments[to_remove] = None

    subMIP_vars = current.model.getVars()

    for var in subMIP_vars:

        if var.getIndex() in to_remove:
            current.x[var] = 0

    return ProblemState(current.x, current.model)


def mutation_op3(current: ProblemState, rnd_state):
    #state = copy.deepcopy(state)

    discrete = find_discrete_index(current)

    to_remove = rnd_state.choice(discrete, size=to_destroy_mut(discrete, delta=0.75))
    assignments = current.solution.copy()
    assignments[to_remove] = None
    sub_vars = current.model.getVars()
    for var in sub_vars:

        if var.getIndex() in to_remove:
            current.x[var] = 0

    return ProblemState(current.x, current.model)


def solve(instance, gap, time, destroy_set=None, var_to_val=None):
    model = scip.Model()
    model.hideOutput()
    model.readProblem(instance)
    model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
    model.setParam("limits/gap", gap)
    model.setParam('limits/time', time)

    if destroy_set:
        for var in model.getVars():
            index = var.getIndex()
            if index not in destroy_set:
                model.addCons(var == var_to_val[var])

    model.optimize()

    var_to_val = {}
    for var in model.getVars():
        var_to_val[var] = model.getVal(var)

    return ProblemState(instance, var_to_val)
