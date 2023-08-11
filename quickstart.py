from alns.select import MABSelector
from alns.accept import HillClimbing
from alns.stop import MaxIterations

# Contextual multi-armed bandits
from mabwiser.mab import LearningPolicy

# Meta-solver built on top of SCIP
from balans.solver import Balans, DestroyOperators, RepairOperators

# Balans
balans = Balans(destroy_ops=[DestroyOperators.Dins, 
                             DestroyOperators.Mutation, 
                             DestroyOperators.Local_Branching,
                             DestroyOperators.Rens, 
                             DestroyOperators.Rins,
                             DestroyOperators.Crossover],
                repair_ops=[RepairOperators.Repair],
                selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=6, num_repair=1,
                                     learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50)),
                accept=HillClimbing(),
                stop=MaxIterations(6))

# Run
result = balans.solve("bin_packing.cip")

# Result
print("Best solution:", result.best_state.solution())
print("Best solution objective:", result.best_state.objective())