#!/usr/bin/env python
# coding: utf-8

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
from base_state import State
import copy


def to_destroy_rins(init_sol, init_sol2, discrete):
    to_remove = np.where(np.in1d(init_sol, init_sol2))[0]
    return to_remove


def lp_relax(state: State):
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

    state = State(solution, state.model)
    print("current iteration: ", state.solution)
    print("current obj val: ", state.objective())

    return state


def rins(state: State, rnd_state):
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

    return State(state.x, state.model)
