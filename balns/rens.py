#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pyscipopt as scip
from balns import BaseOperator
from balns import OperatorExtractor
from utils import MIPState
import pyscipopt as scip

class _Mutation(OperatorExtractor):
    """
    Solution class for the mip problem. It stores the current
    solution as a vector of variables, one for each item.

    Current objective also stored.
    """

    def __init__(self, solution,model,var_features,lp_relaxed_value,init_value,sense1):
        self.model=model
        self.solution=solution
        self.n=len(solution)
        self.var_features=var_features #dataframe
        self.lp_relaxed_value=lp_relaxed_value
        self.init_value=init_value
        self.sense1=sense1
        # print(x)

    def to_destroy(self,discretes) -> int:
        delta=0.25
        #print("type solution",type(self.solution))
        #state = np.array(self.solution)
        #print ("koy", round(int(delta * state.sum())))
        #return int(delta * self.solution.sum())
        return int(delta * len(discretes))


    def find_discretes(self):
        discretes = []
        #print(self.var_features['var_type'])
        for i in range(self.n):
            if self.var_features['var_type'][i] == 0 or self.var_features['var_type'][i] == 1:
                discretes.append(i)
        print("discrete vars",len(discretes))
        return discretes

    def mutation_op(self):
        SEED=42
        rnd_state = np.random.RandomState(SEED)
        #print("type solution", type(self.solution))
        #state = np.array(self.solution)
        probs = self.solution / self.solution.sum()  # Only take 1's, for 0's this prob is =0
        #p = np.random.randint(1, 100, size=self.n)
        discretes = self.find_discretes()
        #to_remove = [i if self.var_features['var_type'][i] == 0 or self.var_features['var_type'][i] == 1 for i in range(self.n)]
        #to_remove = rnd_state.choice(discretes, size=self.to_destroy())

        to_remove = rnd_state.choice(discretes, size=self.to_destroy(discretes))
        #assignments = self.solution.copy()
        #self.model.freeSol(self.model.getBestSol())

        print("solution", self.solution)
        #self.model.freeTransform()
        # solve problem
        self.model.optimize()

        print("solution next",  self.model.getObjVal())

        #assignments = self.model.getBestSol()

        assignments=[]
        for v in self.model.getVars():
            if v.name != "n":
                assignments.append(self.model.getVal(v))
        #print(solution)
        assignments = np.array(assignments)

        for v in to_remove:
            print("v",v)
            assignments[v]= self.solution[v]
            #self.model.setSolVal(assignments, v, self.solution[v])

        #self.model.getObjVal()

        #to_remove = rnd_state.choice(np.arange(self.n), size=self.to_destroy())  # Choose randomly and assign=0

        #assignments = self.solution.copy()
        #assignments[to_remove] = 0
        #print("len remove",len(to_remove))
        cand = MIPState(assignments,self.model)
        print("obj val next", self.model.getObjVal())
        return self.mutation_repair(cand, rnd_state,self.model.getObjVal())

        # return KnapsackState(x=assignments) if delta*x.objective() + (1-delta)*xlp.objective()


    def mutation_repair(self,cand, rnd_state,obj_val):
        #unselected = np.argwhere(cand.solution == 0)
        #rnd_state.shuffle(unselected)

        # while True:
        #     can_insert = w[unselected] <= W - state.weight()
        #     unselected = unselected[can_insert]
        #
        #     if len(unselected) != 0:
        #         insert, unselected = unselected[0], unselected[1:]
        #         state.x[insert] = 1
        #     else:
        #         return cand if lp_relaxed_value >= cand.objective() else lp_sol
        if self.sense1 == "minimize":
            return cand if self.init_value >= obj_val else MIPState(self.solution,self.model)
        else:
            return cand if self.init_value <= obj_val else MIPState(self.solution,self.model)
        #return cand



