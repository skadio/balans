import os
import numpy as np

from alns import ALNS
from alns.accept import *
from alns.select import *
from alns.stop import *

from balans.base_state import _State
from balans.base_instance import _Instance
from balans.mutation import mutation_25
from balans.repair.repair import repair
from balans.utils import Constants
from tests.test_base import BaseTest
from mabwiser.mab import LearningPolicy

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = TEST_DIR + os.sep + ".." + os.sep


class CrossoverTest(BaseTest):

    def test_crossover(self):

        # Set RNG
        np.random.seed(Constants.default_seed)

        # ALNS
        alns = ALNS(np.random.RandomState(Constants.default_seed))

        # Operators
        alns.add_destroy_operator(mutation_25)
        alns.add_repair_operator(repair)

        # Create instance from mip file
        instance_name = "neos-5140963-mincio.mps.gz"
        instance_path = os.path.join(ROOT_DIR, "data", instance_name)
        instance = _Instance(instance_path)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.solve(gap=0.50, time=30)

        # Initial state with the initial solution
        initial_state = _State(instance, initial_index_to_val, initial_obj_val)

        # Bandit selector
        select = MABSelector(scores=[5, 2, 1, 0.5],
                             num_destroy=1,
                             num_repair=1,
                             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))

        # Accept criterion
        accept = HillClimbing()

        # Stop condition
        stop = MaxIterations(5)

        # Run
        result = alns.iterate(initial_state, select, accept, stop)

        # Result
        final_obj_value = result.best_state.objective()
        print("Objective", final_obj_value)

        # Assert
        self.assertTrue(self.is_better(initial_obj_val, final_obj_value, instance.sense))
