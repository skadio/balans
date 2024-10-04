import os
import unittest

from alns.accept import *
from alns.select import *
from alns.stop import *
from mabwiser.mab import LearningPolicy

from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants
from tests.test_base import BaseTest


class CorrectnessTest(BaseTest):
    is_skip = True

    @unittest.skipIf(is_skip, "Skipping correctness 1")
    def test_correctness1(self):
        # Input
        instance = "50v-10.mps"
        instance_path = os.path.join(Constants.DATA_MIP, instance)
        # Balans
        balans = Balans(destroy_ops=[DestroyOperators.Crossover,
                                     DestroyOperators.Dins,
                                     DestroyOperators.Mutation_25,
                                     DestroyOperators.Local_Branching_10,
                                     DestroyOperators.Rins_25,
                                     DestroyOperators.Proximity_05,
                                     DestroyOperators.Rens_25,
                                     DestroyOperators.Random_Objective],
                        repair_ops=[RepairOperators.Repair],
                        selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=8, num_repair=1,
                                             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50)),
                        accept=HillClimbing(),
                        stop=MaxIterations(5),
                        seed=Constants.default_seed)

        # Run
        result = balans.solve(instance_path)
        objective = result.best_state.objective()
        self.assertLess(objective, 4000)  # iter_30 --> 3500
        self.assertGreater(objective, 3311)

    @unittest.skipIf(is_skip, "Skipping correctness 2")
    def test_correctness2(self):
        # Input
        instance = "30n20b8.mps"
        instance_path = os.path.join(Constants.DATA_MIP, instance)
        # Balans
        balans = Balans(destroy_ops=[DestroyOperators.Crossover,
                                     DestroyOperators.Dins,
                                     DestroyOperators.Mutation_25,
                                     DestroyOperators.Local_Branching_10,
                                     DestroyOperators.Rins_25,
                                     DestroyOperators.Proximity_05,
                                     DestroyOperators.Rens_25,
                                     DestroyOperators.Random_Objective],
                        repair_ops=[RepairOperators.Repair],
                        selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=8, num_repair=1,
                                             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50)),
                        accept=HillClimbing(),
                        stop=MaxIterations(5),
                        seed=Constants.default_seed)

        # Run
        result = balans.solve(instance_path)
        objective = result.best_state.objective()
        self.assertLess(objective, 530)  # iter_15 --> 500
        self.assertGreater(objective, 301)

    @unittest.skipIf(is_skip, "Skipping correctness 3")
    def test_correctness3(self):
        # Input
        instance = "b1c1s1.mps"
        instance_path = os.path.join(Constants.DATA_MIP, instance)
        # Balans
        balans = Balans(destroy_ops=[DestroyOperators.Crossover,
                                     DestroyOperators.Dins,
                                     DestroyOperators.Mutation_25,
                                     DestroyOperators.Local_Branching_10,
                                     DestroyOperators.Rins_25,
                                     DestroyOperators.Proximity_05,
                                     DestroyOperators.Rens_25],
                        repair_ops=[RepairOperators.Repair],
                        selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                                             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50)),
                        accept=HillClimbing(),
                        stop=MaxIterations(10),
                        seed=Constants.default_seed)

        # Run
        result = balans.solve(instance_path)
        objective = result.best_state.objective()
        self.assertLess(objective, 68500)  # iter_30 --> 50000
        self.assertGreater(objective, 24544)
