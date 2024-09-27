import os
from alns.accept import *
from alns.select import *
from alns.stop import *
import numpy as np
from alns.ALNS import ALNS

from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants
from tests.test_base import BaseTest
from balans.base_state import _State
from balans.base_instance import _Instance

from mabwiser.mab import LearningPolicy
import pyscipopt as scip


class RinsTest(BaseTest):

    def test_rins_t1(self):
        # Input
        instance = "model.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Rins_25]
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

    def test_rins_t2(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Rins_25]
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

    def test_rins_t3(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = Constants.default_seed

        model = scip.Model()
        model.hideOutput()
        model.readProblem(instance_path)
        instance = _Instance(model)

        index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index_to_val:", index_to_val)

        initial2 = _State(instance, index_to_val, -30)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.RandomState(seed))
        alns.add_destroy_operator(DestroyOperators.Rins_25)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)
        # Retrieve the final solution
        best = result.best_state
        print(f"Best heuristic solution objective is {best.objective()}.")
        # self.assertEqual(result.best_state.objective(), -60)
        self.is_not_worse(-30, result.best_state.objective(), "minimize")

    def test_rins_t4(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = Constants.default_seed

        model = scip.Model()
        model.hideOutput()
        model.readProblem(instance_path)
        instance = _Instance(model)

        index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0},
                          -30, )

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.RandomState(seed))
        alns.add_destroy_operator(DestroyOperators.Rins_25)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(5)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)
        # Retrieve the final solution
        best = result.best_state
        print(f"Best heuristic solution objective is {best.objective()}.")
        self.assertEqual(result.best_state.objective(), -30.0)
