import os

from alns.accept import *
from alns.select import *
from alns.stop import *
from mabwiser.mab import LearningPolicy

from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants
from tests.test_base import BaseTest


class RepairTest(BaseTest):

    BaseTest.mip_solver = Constants.gurobi_solver

    def test_repair(self):

        # Input
        instance = "neos-5140963-mincio.mps"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Mutation_25]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(100)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        # Assert
        self.is_not_worse(balans.initial_obj_val, result.best_state.objective(),
                          balans.instance.mip.org_objective_sense)
