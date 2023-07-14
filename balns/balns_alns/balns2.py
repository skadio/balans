import matplotlib.pyplot as plt
import numpy as np
from mabwiser.mab import LearningPolicy

from alns import ALNS
from alns.accept import *
from alns.select import *
from alns.stop import *

import numpy as np
import pyscipopt as scip
import pandas as pd
from typing import List, Union

from utils import ProblemState
from readinstance import ReadInstance

from abc import abstractmethod
class BaseDestroy:

    @abstractmethod
    def get_index(state:ProblemState):
        pass


    def destroy(state:ProblemState)

        index = self.get_index(state)

        for i in range(index):
            state.solution[idx] = None

        return state


class RensDestroy(BaseDestroy):

    @override
    def get_index(state:ProblemState):
        # TODO

        return index


    def to_destroy(self, discretes) -> int:
        delta = 0.25
        return int(delta * len(discretes))

    def find_discrete(ProblemState: state):
        discrete = []
        for i in range(len(state.x)):
            if self.var_features['var_type'][i] == 0 or self.var_features['var_type'][i] == 1:
                discrete.append(i)
        print("discrete vars", len(discrete))
        return discrete

class MutationDestroy(BaseDestroy):

    @override
    def get_index(ProblemState: state)
        # TODO


        return index

class ProblemState:
    """
    Generic problem class for the mip problem. It stores the current
    solution as a vector of variables, one for each item.

    Current objective also stored.
    """
    def __init__(self, solution, model):
        self.solution = solution #scip.sol()
        self.model = model
        self.incumbent = solution
        self.incumbent_objective = -1
        self.x = self.transform_solution_to_array() #x is the array version of the solution


    #I need some converter from array to solution
    def objective(self):
        #retrieve the objective value of the current solution
        return self.model.getSolObjVal(self.solution)

    def proxy_objective(self) -> int:
        p = np.random.randint(1, 100, size=len(self.x))
        return p @ self.x

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


def initial_state(instance_path,gap,time) -> ProblemState:
    # TODO implement a function that returns an initial solution

    # TODO Solve with scip stop at feasible
    instance = ReadInstance(problem_instance_file=instance_path)
    model = instance.get_model()

    # solution gap is less than %50  > STOP, terrible but, good start.
    model.setParam("limits/gap", gap)
    model.setParam('limits/time', time)
    model.optimize()
    solution = []
    for v in model.getVars():
        if v.name != "n":
            solution.append(model.getVal(v))
    #solution = np.array(solution)
    len_sol = len(solution)
    solution = model.getBestSol()
    #print("init sol", self.model.getObjVal())

    #solution2=model.createSol() #scip in icinde tanimli
    #solution3=model.createSol()
    state = ProblemState(solution, model)

    return state





def destroy_rens(current: ProblemState, rnd_state: rnd.RandomState) -> ProblemState:
    # TODO implement how to destroy the current state, and return the destroyed
    #  state. Make sure to (deep)copy the current state before modifying!

    state = state.copy()

    rens = RensDestroy()

    return rens.destroy(state)


def destroy_rins(current: ProblemState, rnd_state: rnd.RandomState) -> ProblemState:
    # TODO implement how to destroy the current state, and return the destroyed
    #  state. Make sure to (deep)copy the current state before modifying!
    pass


def repair(destroyed: ProblemState, rnd_state: rnd.RandomState) -> ProblemState:

    for i in range(destroyed.solution.len):
        if destroy.solution[i]:
            destroy.model.fix(solution[i])

    solution = model.solve()

    state = ProblemState(solution, destroy.model)

    return state


def repair_only_better(destroyed: ProblemState, rnd_state: rnd.RandomState) -> ProblemState:

    for i in range(destroyed.solution.len):
        if destroy.solution[i]:
            destroy.model.fix(solution[i])

    solution = model.solve()

    state = ProblemState(solution, destroy.model)

    return state


# Create the initial solution
init_sol = initial_state(instance)
print(f"Initial solution objective is {init_sol.objective()}.")

# Create ALNS and add one or more destroy and repair operators
alns = ALNS(rnd.RandomState(seed=42))
alns.add_destroy_operator(destroy_rens)
alns.add_destroy_operator(destroy_mutation)


alns.add_destroy_operator(destroy_rins)
alns.add_destroy_operator(destroy_mutation)
alns.add_destroy_operator(destroy)
alns.add_destroy_operator(destroy)
alns.add_repair_operator(repair)
alns.add_repair_operator(repair_only_better)


# Configure ALNS
mab = MAB(arms, LearningPolicy.LinTS(alpha=1.25))
select = MABSelector(mab, num_destroy=8, num_repair=2, reward[1, 2, 3, 4])
accept = HillClimbing()
stop = MaxRuntime(60)

# Run the ALNS algorithm
result = alns.iterate(init_sol, select, accept, stop)

# Retrieve the final solution
best = result.best_state
print(f"Best heuristic solution objective is {best.objective()}.")