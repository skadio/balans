import os
import unittest
import numpy as np

from alns import ALNS
from alns.accept import *
from alns.select import *
from alns.stop import *

from balns.utils import Constants

from mabwiser.mab import LearningPolicy, NeighborhoodPolicy


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = TEST_DIR + os.sep + ".." + os.sep


class BALNSTest(unittest.TestCase):

    def test_balns(self):

        # Set RNG
        np.random.seed(Constants.default_seed)

        # ALNS
        alns = ALNS(np.random.RandomState(Constants.default_seed))

        # Operators
        alns.add_destroy_operator(mutation)

        # Initial solution
        instance = "neos-5140963-mincio.mps.gz"
        instance_path = os.path.join(ROOT_DIR, "data", instance)
        initial_state = solve(instance_path, gap=0.50, time=30)

        initial_solution_value = initial_state.objective()

        # Bandit selector
        select = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=5, num_repair=1,
                             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))

        # Accept criterion
        accept = HillClimbing()

        # Stop condition
        stop = MaxIterations(5)

        # Run
        result = alns.iterate(initial_state, select, accept, stop)

        # Result
        print(f"Found solution with objective {result.best_state.objective()}.")

        # Check optimization sense first (this assumes minimization)
        self.assertTrue(result.best_state.objective() < initial_solution_value)
