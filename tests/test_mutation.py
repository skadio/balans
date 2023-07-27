import os
from alns.accept import *
from alns.select import *
from alns.stop import *
from pyscipopt import Model
import numpy as np
from alns.ALNS import ALNS

from balans.destroy import DestroyOperators
from balans.repair import RepairOperators
from balans.solver import Balans
from balans.utils import Constants
from tests.test_base import BaseTest
from balans.base_state import _State
from balans.base_instance import _Instance

from mabwiser.mab import LearningPolicy, NeighborhoodPolicy

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = TEST_DIR + os.sep + ".." + os.sep



class MutationTest(BaseTest):

    def test_mutation(self):
        # Input
        instance = "noswot.mps.gz"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Mutation]
        repair_ops = [RepairOperators.Repair]

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(45)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        # Assert
        self.assertIsBetter(balans.initial_obj_val, result.best_state.objective(), balans.instance.sense)

    def test_mutation_t1(self):
        # Input
        instance = "model.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Mutation]
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

    def test_mutation_t2(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Mutation2]
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
        self.assertEqual(result.best_state.objective(), -60.0)

    def test_mutation_t3(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Mutation2]
        repair_ops = [RepairOperators.Repair]

        instance = _Instance(instance_path)

        var_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial var to val:", var_to_val)

        initial2 = _State(instance, {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0},
                          -30,
                          lp_var_to_val={1: 60.0, 0: 0.0, 4: 0.0, 3: 0.0, 2: 0.0},
                          lp_obj_val=-60.0)

        # Initial solution
        initial_var_to_val, initial_obj_val, lp_var_to_val, lp_obj_val \
            = instance.solve(is_initial_solve=True)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS()
        alns.add_destroy_operator(DestroyOperators.Mutation2)
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
        #self.assertEqual(result.best_state.objective(), -60)
        self.assertIsBetter(-30, result.best_state.objective(), "minimize")

    def test_mutation_t4(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR, instance)

        # Parameters
        seed = Constants.default_seed

        destroy_ops = [DestroyOperators.Mutation2]
        repair_ops = [RepairOperators.Repair]

        instance = _Instance(instance_path)

        var_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial var to val:", var_to_val)

        initial2 = _State(instance, {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0},
                          -30,
                          lp_var_to_val={1: 60.0, 0: 0.0, 4: 0.0, 3: 0.0, 2: 0.0},
                          lp_obj_val=-60.0)

        # Initial solution
        initial_var_to_val, initial_obj_val, lp_var_to_val, lp_obj_val \
            = instance.solve(is_initial_solve=True)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS()
        alns.add_destroy_operator(DestroyOperators.Mutation2)
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
        self.assertEqual(result.best_state.objective(), -60.0)
