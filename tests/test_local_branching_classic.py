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


class LocalBranchingTest(BaseTest):

    def test_local_branching_classic_t1(self):
        # Input
        instance = "test5.13.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR, instance)

        # Parameters
        seed = 123456

        instance = _Instance(instance_path)

        index_to_val = {0: 0.0, 1: 0.0, 2: 0.0, 3: 10.0, 4: 10.0, 5: 20.0, 6: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, index_to_val, -40)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.solve(is_initial_solve=True)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.RandomState(seed))
        alns.add_destroy_operator(DestroyOperators.Local_Branching4)
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
        self.assertEqual(result.best_state.objective(), -61.0)

    def test_local_branching_classic_t2(self):
        # Input
        instance = "test5.13.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR, instance)

        # Parameters
        seed = 123456

        instance = _Instance(instance_path)

        index_to_val = {0: 1.0, 1: 1.0, 2: 0.0, 3: 10.0, 4: 10.0, 5: 20.0, 6: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, index_to_val, -42)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.solve(is_initial_solve=True)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.RandomState(seed))
        alns.add_destroy_operator(DestroyOperators.Local_Branching4)
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
        self.assertEqual(result.best_state.objective(), -63.0)

    def test_local_branching_v2_t3(self):
        # Input
        instance = "test5.13.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR, instance)

        # Parameters
        seed = 123456

        instance = _Instance(instance_path)

        index_to_val = {0: 1.0, 1: 0.0, 2: 0.0, 3: 10.0, 4: 10.0, 5: 20.0, 6: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, index_to_val, -41)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.solve(is_initial_solve=True)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.RandomState(seed))
        alns.add_destroy_operator(DestroyOperators.Local_Branching4)
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
        self.assertEqual(result.best_state.objective(), -62.0)
