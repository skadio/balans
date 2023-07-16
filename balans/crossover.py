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
from base_state import State
from readinstance import ReadInstance


def crossover(state: State, rnd_state):
    sub_vars = state.model.getVars()
    same_vars = []
    for var in sub_vars:
        if init_sol.x[var] == init_sol2.x[var]:
            same_vars.append(var)

    for var in same_vars:
        state.x[var] = 0

    return State(state.x, state.model)
