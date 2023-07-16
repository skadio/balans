import numpy as np
import pyscipopt as scip
from balans import BaseOperator
from balans import OperatorExtractor
from utils import MIPState
import pyscipopt as scip


def local_branching(state: State, rnd_state):
    sub_vars = state.model.getVars()
    same_vars = []
    for var in sub_vars:
        if init_sol.x[var] == init_sol2.x[var]:
            same_vars.append(var)

    for var in same_vars:
        state.x[var] = 0

    return State(state.x, state.model)
