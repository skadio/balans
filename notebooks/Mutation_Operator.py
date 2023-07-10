#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pyscipopt as scip

import pandas as pd
from typing import List, Union


SEED = 42
np.random.seed(SEED)


class BaseOperator:
    def __init__(self, problem_instance_file: str) -> None:
        self.model = scip.Model()
        self.model.hideOutput()
        self.model.readProblem(problem_instance_file)


class OperatorExtractor(BaseOperator):
    def __init__(self, problem_instance_file: str) -> None:
        super().__init__(problem_instance_file)

    def LP_relax(self):

        """
        Gets and Solves LP relaxed version of the same problem

        Returns
        -------
        objective value=float
        solution =array
        len_sol=int
        """
        # Extract variable and constraints features
        for v in self.model.getVars():
            self.model.chgVarType(v, 'CONTINUOUS') #Continious relaxation of the problem
        self.model.optimize()
        solution=[]
        for v in self.model.getVars():
            if v.name != "n":
                solution.append(self.model.getVal(v))
        #print(solution)
        len_sol = len(solution)
        return self.model.getObjVal(),solution,len_sol
    
    
    def get_sense(self) -> str:
        sense = self.model.getObjectiveSense()
        return sense

    
    def Model(self):
        return self.model
    
    def extract_feature(self) -> Union[List[Union[int, float]], np.ndarray, pd.Series, pd.DataFrame]:
        """
        Extracts features from MIP instances

        Returns
        -------
        feature_features : Union[List[Union[int, float]], np.ndarray, pd.Series, pd.DataFrame]
            The extracted features
        """
        # Extract variable and constraints features
        variable_features = self.extract_variable_features()
        constraint_features, constraint_signs = self.extract_constraint_features()
        objective_sense = self.get_sense()

        # Create a DataFrame with the desired format for MABSelector
        feature_df = pd.concat([variable_features, constraint_features, constraint_signs], axis=1)
        feature_df.insert(0, 'objective_sense', objective_sense)
        feature_df.loc[1:, 'objective_sense'] = np.nan
        feature_df.fillna('nan', inplace=True)
        feature_df.reset_index(drop=True, inplace=True)

        return feature_df


import os


def run_mip_operator_extractor(instance_path):

    
    operator_extractor = OperatorExtractor(problem_instance_file=instance_path)    
    
    lp_relaxed_value, solution,n=operator_extractor.LP_relax()
    model=operator_extractor.Model()

    print("LP RELAXED SOL IS:", solution)
    print("Num Var:", n)
    
    return lp_relaxed_value, solution, n,model
if __name__ == "__main__":

    instance_path = "data/neos-5140963-mincio.mps.gz"

    # Create MIP instance and LP relaxed Solution
    lp_relaxed_value, solution, n,model=run_mip_operator_extractor(instance_path)
    

# # MIP State Class 
# ##Includes current objective and problem parameters.



class MIPState:
    """
    Solution class for the mip problem. It stores the current
    solution as a vector of variables, one for each item.
    
    Current objective also stored.
    """

    def __init__(self, x: np.ndarray):
        self.x = x
        #print(x)

    def objective(self) -> int:
        #return model.getSolObjVal(self.x)
        return model.getObjVal(self.x)
    
    def proxy_objective(self) -> int:
        #return model.getSolObjVal(self.x)
        p = np.random.randint(1, 100, size=n)
        return p @ self.x
    
    def lenght(self)-> int:
        return len(self.x)
    
    #def objective2(self) -> int:
        #return model.getObjective()


# # Read the LP solution

lp_sol = MIPState(solution)
lp_sol.objective()
# Terrible - but simple - first solution, where only the first item is
# selected.
#init_sol = MIPState(np.zeros(n))
#init_sol.x[0] = 1

#init_sol.objective()


# # Mutation Operator
rnd_state = np.random.RandomState(SEED)

#print(rnd_state)
#In our operators this is delta

# Percentage of items to remove in each iteration
delta = .25 #change this to delta
n=196

def to_destroy(state: MIPState) -> int:
    return int(delta * state.x.sum())


def mutation(state: MIPState, rnd_state):
    state.x=np.array(state.x)
    probs = state.x / state.x.sum() #Only take 1's, for 0's this prob is =0
    p = np.random.randint(1, 100, size=n)
    #to_remove = rnd_state.choice(np.arange(n), size=to_destroy(state), p=probs) #vectordeki bazilarini secip 0 liyoruz.
    to_remove = rnd_state.choice(np.arange(n), size=to_destroy(state)) #Choose randomly and assign=0

    assignments = state.x.copy() #copy leyip store ediyoruz.
    assignments[to_remove] = 0 #secilenleri yenisinde store edip

    cand= MIPState(x=assignments)
    return mutation_repair(cand,rnd_state)

    #return KnapsackState(x=assignments) if delta*x.objective() + (1-delta)*xlp.objective()
    
def mutation_repair(cand, rnd_state):
    unselected = np.argwhere(cand.x == 0)
    rnd_state.shuffle(unselected)
    """
    while True:
        can_insert = w[unselected] <= W - state.weight()
        unselected = unselected[can_insert]

        if len(unselected) != 0:
            insert, unselected = unselected[0], unselected[1:]
            state.x[insert] = 1
        else:
            return cand if lp_relaxed_value >= cand.objective() else lp_sol
    """
    return cand if lp_relaxed_value >= cand.proxy_objective() else lp_sol


mut_sol = MIPState(solution)
#lp_sol.objective()
mut_sol2 = mutation(mut_sol,rnd_state)
mut_sol2
# Terrible - but simple - first solution, where only the first item is
# selected.
#init_sol = MIPState(np.zeros(n))
#init_sol.x[0] = 1

#init_sol.objective()






