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

class CorrectnessTest(BaseTest):

    def test_correctness1(self):
        # Input
        instance = "50v-10.mps"
        instance_path = os.path.join(Constants.DATA_MIP, instance)
        # Balans
        balans = Balans(destroy_ops=[DestroyOperators.Crossover,
                                     DestroyOperators.Dins,
                                     DestroyOperators.Mutation,
                                     DestroyOperators.Local_Branching,
                                     DestroyOperators.Rins,
                                     DestroyOperators.Proximity,
                                     DestroyOperators.Rens,
                                     DestroyOperators.Random_Objective],
                        repair_ops=[RepairOperators.Repair],
                        selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=8, num_repair=1,
                                             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50)),
                        accept=HillClimbing(),
                        stop=MaxIterations(30),
                        seed=Constants.default_seed)

        # Run
        result = balans.solve(instance_path)
        objective = result.best_state.objective()
        self.assertLess(objective, 3500)
        self.assertGreater(objective, 3311)

    def test_correctness2(self):
        # Input
        instance = "30n20b8.mps"
        instance_path = os.path.join(Constants.DATA_MIP, instance)
        # Balans
        balans = Balans(destroy_ops=[DestroyOperators.Crossover,
                                     DestroyOperators.Dins,
                                     DestroyOperators.Mutation,
                                     DestroyOperators.Local_Branching,
                                     DestroyOperators.Rins,
                                     DestroyOperators.Proximity,
                                     DestroyOperators.Rens,
                                     DestroyOperators.Random_Objective],
                        repair_ops=[RepairOperators.Repair],
                        selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=8, num_repair=1,
                                             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50)),
                        accept=HillClimbing(),
                        stop=MaxIterations(15),
                        seed=Constants.default_seed)

        # Run
        result = balans.solve(instance_path)
        objective = result.best_state.objective()
        self.assertLess(objective, 500)
        self.assertGreater(objective, 301)

    def test_correctness3(self):
        # Input
        instance = "b1c1s1.mps"
        instance_path = os.path.join(Constants.DATA_MIP, instance)
        # Balans
        balans = Balans(destroy_ops=[DestroyOperators.Crossover,
                                     DestroyOperators.Dins,
                                     DestroyOperators.Mutation,
                                     DestroyOperators.Local_Branching,
                                     DestroyOperators.Rins,
                                     DestroyOperators.Proximity,
                                     DestroyOperators.Rens],
                        repair_ops=[RepairOperators.Repair],
                        selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                                             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50)),
                        accept=HillClimbing(),
                        stop=MaxIterations(30),
                        seed=Constants.default_seed)

        # Run
        result = balans.solve(instance_path)
        objective = result.best_state.objective()
        self.assertLess(objective, 50000)
        self.assertGreater(objective, 24544)
