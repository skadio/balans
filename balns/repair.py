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


def repair_op(current: ProblemState, rnd_state) -> ProblemState:

    current = solve(current.instance, current.destroy_set, current.var_to_val)

    return current
