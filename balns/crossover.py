#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pyscipopt as scip
from balns import BaseOperator
from balns import OperatorExtractor
from utils import MIPState
import pyscipopt as scip

class _Crossover(OperatorExtractor):
    """
    Solution class for the mip problem. It stores the current
    solution as a vector of variables, one for each item.

    Current objective also stored.
    """

    def __init__(self, solution,solution2,model,var_features,lp_relaxed_value,init_value,init_value2,sense1):
        self.model=model
        self.solution=solution
        self.solution2 = solution2
        self.n=len(solution)
        self.var_features=var_features #dataframe
        self.lp_relaxed_value=lp_relaxed_value
        self.init_value=init_value
        self.init_value2=init_value2
        self.sense1=sense1
        # print(x)

    def to_destroy(self) -> int:

        to_remove=np.where(np.in1d(self.solution, self.solution2))[0]

        return to_remove


    def find_discretes(self):
        discretes = []
        #print(self.var_features['var_type'])
        for i in range(self.n):
            if self.var_features['var_type'][i] == 0 or self.var_features['var_type'][i] == 1:
                discretes.append(i)
        print("discrete vars",len(discretes))
        return discretes

    def crossover_op(self):

        #get the same ones
        to_remove = self.to_destroy()


        print("solution", self.solution)

        self.model.optimize()

        print("solution next",  self.model.getObjVal())

        assignments=[]
        for v in self.model.getVars():
            if v.name != "n":
                assignments.append(self.model.getVal(v))
        #print(solution)
        assignments = np.array(assignments)

        for v in to_remove:
            assignments[v]= self.solution[v]

        cand = MIPState(assignments,self.model)
        print("obj val next", self.model.getObjVal())
        return self.crossover_repair(cand,self.model.getObjVal())



    def crossover_repair(self,cand,obj_val):

        if self.sense1 == "minimize":
            return cand if self.init_value >= obj_val else MIPState(self.solution,self.model)
        else:
            return cand if self.init_value <= obj_val else MIPState(self.solution,self.model)
        #return cand



