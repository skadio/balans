import os
from alns.accept import *
from alns.select import *
from alns.stop import *

from balans.destroy import DestroyOperators
from balans.repair import RepairOperators
from balans.solver import Balans
from balans.utils import Constants
from tests.test_base import BaseTest

from mabwiser.mab import LearningPolicy, NeighborhoodPolicy


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = TEST_DIR + os.sep + ".." + os.sep


class RepairTest(BaseTest):

    def test_repair(self):

        # Input
        instance = "neos-5140963-mincio.mps.gz"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Mutation2]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(100)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        # Assert
        self.assertIsBetter(balans.initial_obj_val, result.best_state.objective(), balans.instance.sense)
