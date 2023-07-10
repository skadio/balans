#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pyscipopt as scip
from balns import BaseOperator
from balns import OperatorExtractor
from utils import MIPState


class _Crossover(OperatorExtractor):
    """
    Crossover OperatorS class for the mip problem. It stores the current
    solution as a vector of variables, one for each item.

    Current objective also stored.

    Return: new solution
    """

    def __init__(self, solution, solution2, model, var_features, lp_relaxed_value, init_value, init_value2, sense1):
        self.model = model
        self.solution = solution
        self.solution2 = solution2
        self.n = len(solution)
        self.var_features = var_features  # dataframe
        self.lp_relaxed_value = lp_relaxed_value
        self.init_value = init_value
        self.init_value2 = init_value2
        self.sense1 = sense1

    def to_destroy(self) -> int:

        to_remove = np.where(np.in1d(self.solution, self.solution2))[0]

        return to_remove

    def find_discrete(self):
        discrete = []
        for i in range(self.n):
            if self.var_features['var_type'][i] == 0 or self.var_features['var_type'][i] == 1:
                discrete.append(i)
        return discrete

    def crossover_op(self):

        # get the same ones
        to_remove = self.to_destroy()

        self.model.optimize()

        assignments = []
        for v in self.model.getVars():
            if v.name != "n":
                assignments.append(self.model.getVal(v))
        assignments = np.array(assignments)

        for v in to_remove:
            assignments[v] = self.solution[v]

        candidate = MIPState(assignments, self.model)
        return self.crossover_repair(candidate, self.model.getObjVal())

    def crossover_repair(self, candidate, obj_val):

        if self.sense1 == "minimize":
            return candidate if self.init_value >= obj_val else MIPState(self.solution, self.model)
        else:
            return candidate if self.init_value <= obj_val else MIPState(self.solution, self.model)
