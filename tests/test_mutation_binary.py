import os
from alns.accept import *
from alns.select import *
from alns.stop import *

from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants
from tests.test_base import BaseTest
from balans.base_instance import _Instance

from mabwiser.mab import LearningPolicy


class MutationBinaryTest(BaseTest):

    def test_mutation_binary_t1(self):
        # Input
        instance = "model.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Mutation_Binary]
        repair_ops = [RepairOperators.Repair]

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))

        accept = HillClimbing()
        stop = MaxIterations(1)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        self.assertEqual(result.best_state.objective(), 4)

    def test_mutation_binary_t2(self):
        # Input
        instance = "test5.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Mutation_Binary]
        repair_ops = [RepairOperators.Repair]

        instance = _Instance(instance_path)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        # Assert
        self.assertEqual(result.best_state.objective(), -60)
