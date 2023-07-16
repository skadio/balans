from mabwiser.mab import LearningPolicy
import numpy as np
import pyscipopt as scip
from problemstate import ProblemState
from alns import ALNS
from alns.accept import *
from alns.select import *
from alns.stop import *
from mutation import mutation_op, mutation_op2, mutation_op3
from cross import crossover_op
from rins import rins_op
from repair import  repair_op


SEED = 42
np.random.seed(SEED)


def solve(instance, gap, time, destroy_set=None, var_to_val=None):
    model = scip.Model()
    model.hideOutput()
    model.readProblem(instance)
    model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
    model.setParam("limits/gap", gap)
    model.setParam('limits/time', time)

    if destroy_set:
        for var in model.getVars():
            index = var.getIndex()
            if index not in destroy_set:
                model.addCons(var == var_to_val[var])

    model.optimize()

    var_to_val = {}
    for var in model.getVars():
        var_to_val[var] = model.getVal(var)

    return ProblemState(instance, var_to_val)


if __name__ == "__main__":

    # ALNS
    alns = ALNS(np.random.RandomState(SEED))

    # Operators
    alns.add_destroy_operator(mutation_op)
    alns.add_destroy_operator(mutation_op2)
    alns.add_destroy_operator(mutation_op3)
    alns.add_destroy_operator(crossover_op)
    alns.add_destroy_operator(rins_op)
    alns.add_repair_operator(repair_op)

    # Initial solution
    initial_state = solve(instance="neos-5140963-mincio.mps.gz", gap=0.50, time=30)

    # MABSelector
    select = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=5, num_repair=1,
                         learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))

    # Run
    result = alns.iterate(initial_state, select, accept=HillClimbing(), MaxIterations(5))

    print(f"Found solution with objective {result.best_state.objective()}.")
