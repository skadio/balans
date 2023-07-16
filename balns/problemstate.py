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


class ProblemState:
    """
    Generic problem class for the mip problem. It stores the current
    solution as a vector of variables, one for each item.

    Current objective also stored.
    """

    def __init__(self, instance, var_to_val):
        self.instance = instance
        self.var_to_val = var_to_val
        self.destroy_set = None

        self.model = scip.Model()
        self.incumbent = self.var_to_val
        self.x = None

    def objective(self):
        # retrieve the objective value of the current solution
        return self.model.getSolObjVal(self.x)

    def proxy_objective(self) -> int:
        p = np.random.randint(1, 100, size=len(self.solution))
        return p @ self.model

    def length(self) -> int:
        return len(self.solution)

    def get_solution(self):
        return self.solution

    def get_objective(self):
        # TODO output as minimize
        return self.model.getSolObjVal(self.x)

    # for now leave it empty
    def get_context(self):
        return None

    def array_to_sol(self):
        scip_sol = self.model.createSol()
        i = 0
        for var in self.model.getVars():
            # self.model.setSolVal(scip_sol, var, self.solution[var.getIndex()])  # needs to be index
            # self.model.setSolVal(scip_sol, var, self.solution[var.getIndex()])  # needs to be index

            scip_sol[var] = self.solution[var.getIndex()]

        return scip_sol

    def transform_solution_to_array(self):
        solution_array = []
        for i in range(self.model.getNVars()):
            solution_array.append(self.model.getSolVal(self.x, self.model.getVars()[i]))
        solution_array = np.array(solution_array)

        return solution_array

    def transform_solution_to_array2(self):
        solution_array = np.zeros(self.model.getNVars())
        for var in self.model.getVars():
            i = var.getIndex()
            solution_array[i] = self.x[var]
        solution_array = np.array(solution_array)

        return solution_array
