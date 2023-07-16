#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import copy
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

instance_path = "neos-5140963-mincio.mps.gz"

# Terrible - but simple - two first solution, where only the first item is
# selected.
instance = ReadInstance(instance_file=instance_path)
instance2 = ReadInstance(instance_file=instance_path)

# Time =30 and gap limit = 50 percent gap within the solution
init_sol = instance.initial_state(0.50, 30)
# Time =30 and gap limit = 75 percent gap within the solution
init_sol2 = instance2.initial_state(0.75, 30)


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


def find_discrete(state: ProblemState):
    discrete = []
    for i in range(state.length()):
        var_features = extract_variable_features(state)
        if var_features['var_type'][i] == 0 or var_features['var_type'][i] == 1:
            discrete.append(i)
    return discrete


def crossover_op(state: ProblemState, rnd_state):
    sub_vars = state.model.getVars()
    same_vars = []
    for var in sub_vars:
        if init_sol.x[var] == init_sol2.x[var]:
            same_vars.append(var)

    for var in same_vars:
        state.x[var] = 0

    return ProblemState(state.x, state.model)
