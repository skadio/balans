# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyscipopt as scip
from balns import OperatorExtractor

class ProblemState:
    """
    Generic problem class for the mip problem. It stores the current
    solution as a vector of variables, one for each item.

    Current objective also stored.
    """
    def __init__(self, solution, model):
        self.solution = solution
        self.model = model
        self.incumbent = solution
        self.incumbent_objective = -1

    def objective(self):
        #retrieve the objective value of the current solution
        return self.model.getSolObjVal(self.solution)

    def proxy_objective(self) -> int:
        p = np.random.randint(1, 100, size=len(self.solution))
        return p @ self.model

    def length(self) -> int:
        return len(self.model)

    def get_solution(self):
        return self.solution

    def get_objective(self):
        # TODO output as minimize
        return self.model.getObjVal(self.solution)

    #for now leave it empty
    def get_context(self):
        return None

    def transform_solution_to_array(self):
        solution_array = []
        for i in range(self.model.getNVars()):
            solution_array.append(self.model.getSolVal(self.solution, self.model.getVars()[i]))
        solution_array = np.array(solution_array)

        return solution_array

    # def copy(self):
    #     return ProblemState(copy.deepcopy(self.solution), copy.deepcopy(self.model))
