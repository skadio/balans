#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pyscipopt as scip
from balns import BaseOperator
from balns import OperatorExtractor
from utils import MIPState


class _Mutation(OperatorExtractor):
    """
    Mutation Operator class for the mip problem. It stores the current
    solution as a vector of variables, one for each item.

    Current objective also stored.

    Return: new solution
    """

    def __init__(self, solution, model, var_features, lp_relaxed_value, init_value, sense1):
        self.model = model
        self.solution = solution
        self.n = len(solution)
        self.var_features = var_features
        self.lp_relaxed_value = lp_relaxed_value
        self.init_value = init_value
        self.sense1 = sense1

    def to_destroy(self, discretes) -> int:
        delta = 0.25
        return int(delta * len(discretes))

    def find_discrete(self):
        discrete = []
        for i in range(self.n):
            if self.var_features['var_type'][i] == 0 or self.var_features['var_type'][i] == 1:
                discrete.append(i)
        print("discrete vars", len(discrete))
        return discrete

    def mutation_op(self):
        SEED = 42
        rnd_state = np.random.RandomState(SEED)

        discrete = self.find_discrete()

        to_remove = rnd_state.choice(discrete, size=self.to_destroy(discrete))
        # solve problem
        self.model.optimize()

        assignments = []
        for v in self.model.getVars():
            if v.name != "n":
                assignments.append(self.model.getVal(v))
        assignments = np.array(assignments)

        for v in to_remove:
            assignments[v] = self.solution[v]

        candidate = MIPState(assignments, self.model)

        return self.mutation_repair(candidate, self.model.getObjVal())

    def mutation_repair(self, candidate, obj_val):
        if self.sense1 == "minimize":
            return candidate if self.init_value >= obj_val else MIPState(self.solution, self.model)
        else:
            return candidate if self.init_value <= obj_val else MIPState(self.solution, self.model)
