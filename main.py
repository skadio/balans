import os

from alns.select import MABSelector
from alns.accept import HillClimbing
from alns.stop import MaxIterations, MaxRuntime
from pyscipopt import Model

# Contextual multi-armed bandits
from mabwiser.mab import LearningPolicy

# Meta-solver built on top of SCIP
from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants

# Balans
balans = Balans(destroy_ops=[DestroyOperators.Crossover,
                             DestroyOperators.Dins,
                             DestroyOperators.Mutation,
                             DestroyOperators.Local_Branching,
                             DestroyOperators.Rens,
                             DestroyOperators.Rins,
                             DestroyOperators.Proximity,
                             DestroyOperators.Zero_Objective],
                repair_ops=[RepairOperators.Repair],
                selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=8, num_repair=1,
                                     learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50)),
                accept=HillClimbing(),
                # stop=MaxIterations(10)
                stop=MaxRuntime(30))

instance = "air05.mps"
instance_path = os.path.join(Constants.DATA_MIP, instance)

# # Run
result = balans.solve(instance_path)
print("Best solution:", result.best_state.solution())
print("Best solution objective:", result.best_state.objective())


# Check for optimality
# model = Model("scip")
# model.readProblem(instance_path)
# model.optimize()
# solution = model.getBestSol()
# print(solution)
# print("Optimal value:", model.getObjVal())
