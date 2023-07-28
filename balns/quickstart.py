import os
# Contextual multi-armed bandits
from mabwiser.mab import LearningPolicy
import numpy as np
# Adaptive large neigborhood
from alns.select import MABSelector
from alns.accept import HillClimbing, RandomWalk
from alns.stop import MaxIterations

from balans.utils import Constants
# Meta-solver built on top of SCIP
from balans.solver import Balans
from balans.destroy import DestroyOperators
from balans.repair import RepairOperators

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = TEST_DIR + os.sep + ".." + os.sep
SEED = 42
np.random.seed(SEED)
rnd_state = np.random.RandomState(SEED)

# Balans
balans = Balans(destroy_ops=[DestroyOperators.Mutation2],
                repair_ops=[RepairOperators.Repair],
                selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                                     learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15)),
                accept=HillClimbing(),
                stop=MaxIterations(50))

# Input
instance = "neos-5140963-mincio.mps.gz"
#instance = "test2.5.cip"
#instance = "test5.5.cip"
instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR, instance)

# Run
result = balans.solve(instance_path)

# result=balans.solve("noswot.mps.gz")

# Result
print("Best solution:", result.best_state.objective())


