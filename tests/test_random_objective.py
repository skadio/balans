import os
from alns.accept import *
from alns.select import *
from alns.stop import *

from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants
from tests.test_base import BaseTest
from balans.base_state import _State
from balans.base_instance import _Instance

from mabwiser.mab import LearningPolicy


class RandomObjectiveTest(BaseTest):

    def test_random_objective_t1(self):
        # Input
        instance = "model.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Random_Objective]
        repair_ops = [RepairOperators.Repair]

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))

        accept = HillClimbing()
        stop = MaxIterations(10)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        self.assertEqual(result.best_state.objective(), 4)

    def test_random_objective_t2(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Random_Objective]
        repair_ops = [RepairOperators.Repair]

        instance = _Instance(instance_path)

        index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", index_to_val)
        obj_value = -40

        initial2 = _State(instance, index_to_val, -30)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(10)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        # Assert
        self.assertEqual(result.best_state.objective(), -60)

    def test_random_objective_with_warm_start(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Random_Objective]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed)

        # Indexes 0, 1, 2 are discrete so only these indexes can be destroyed
        # With this seed, in the firs iteration index=1 is destroy
        # Hence var0 and var2 must remain fixed and only the other variables can change
        # Objective in the next iteration is 50 (minus since sense is minimization)
        initial_index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", initial_index_to_val)

        # Run
        result = balans.solve(instance_path, initial_index_to_val)

        # Retrieve the final solution
        best_state = result.best_state
        best_objective = best_state.objective()
        best_solution = best_state.solution()

        print("Best solution:", result.best_state.solution())
        print("Best solution value:", result.best_state.objective())

