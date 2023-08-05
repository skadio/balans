import os
from alns.accept import *
from alns.select import *
from alns.stop import *
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


class CrossoverTest(BaseTest):

    def test_crossover(self):
        # Input
        instance = "noswot.mps.gz"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR_MIP, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Crossover3]
        repair_ops = [RepairOperators.Repair]

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(5)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        # Assert
        self.assertIsBetter(balans.initial_obj_val, result.best_state.objective(), balans.instance.sense)

    def test_crossover_t1(self):
        # Input
        instance = "model.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR_TOY, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Crossover3]
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

    def test_crossover_t2(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR_TOY, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Crossover3]
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

    def test_crossover_t3(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR_TOY, instance)

        # Parameters
        seed = 123456

        destroy_ops = [DestroyOperators.Crossover3]
        repair_ops = [RepairOperators.Repair]

        instance = _Instance(instance_path)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.solve(is_initial_solve=True)

        # Indexes 0, 1, 2 are discrete so only these indexes can be destroyed
        # With this seed, in the firs iteration index=1 is destroy
        # Hence var0 and var2 must remain fixed and only the other variables can change
        # Objective in the next iteration is 50 (minus since sense is minimization)
        initial_index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", initial_index_to_val)

        initial2 = _State(instance, initial_index_to_val, -30)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.RandomState(seed))
        alns.add_destroy_operator(DestroyOperators.Crossover3)
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
        self.assertIsBetter(-30, result.best_state.objective(), "minimize")

    def test_crossover_t4(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR_TOY, instance)

        # Parameters
        seed = 123456

        destroy_ops = [DestroyOperators.Crossover3]
        repair_ops = [RepairOperators.Repair]

        instance = _Instance(instance_path)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.solve(is_initial_solve=True)

        # Indexes 0, 1, 2 are discrete so only these indexes can be destroyed
        # With this seed, in the firs iteration index=1 is destroy
        # Hence var0 and var2 must remain fixed and only the other variables can change
        # Objective in the next iteration is 50 (minus since sense is minimization)
        initial_index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", initial_index_to_val)

        initial2 = _State(instance, initial_index_to_val, -30)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.RandomState(seed))
        alns.add_destroy_operator(DestroyOperators.Crossover3)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)

        # Retrieve the final solution
        best_state = result.best_state
        best_objective = best_state.objective()
        best_solution = best_state.solution()

        print(f"Best heuristic solution objective is {best_objective}.")
        self.assertEqual(best_objective, -60.0)

    def test_crossover_t4_with_warm_start(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR_TOY, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Crossover3]
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


        # Assert objective
        self.assertEqual(best_objective, -60.0)
