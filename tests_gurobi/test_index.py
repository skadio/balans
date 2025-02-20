import os

import numpy as np
from alns.accept import *
from alns.select import *
from alns.stop import *
from mabwiser.mab import LearningPolicy
import gurobipy as grb
from gurobi_onboarder import init_gurobi

from balans.base_instance import _Instance
from balans.base_mip import create_mip_solver
from balans.base_state import _State
from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants
from tests.test_base import BaseTest

SEED = 42
np.random.seed(SEED)
rnd_state = np.random.RandomState(SEED)


class IndexTest(BaseTest):

    def test_cont_index_t1(self):
        # Testing whether we get the correct index set

        # Input
        instance = "neos-5140963-mincio.mps"
        instance_path = os.path.join(Constants.DATA_MIP, instance)

        # Initialize Gurobi model

        gurobi_venv, GUROBI_FOUND = init_gurobi.initialize_gurobi()
        model = grb.read(instance_path,env=gurobi_venv)
        model.setParam("OutputFlag", 0)
        variables = model.getVars()

        # Set discrete indexes
        discrete = []
        for var in variables:
            if var.VType == grb.GRB.CONTINUOUS:
                discrete.append(var.index)

        non_destroy_size = int(0.5 * len(discrete))

        non_destroy_set = set(rnd_state.choice(discrete, non_destroy_size))

        self.assertEqual(non_destroy_set, {3, 6, 7, 10, 12})

    def test_disc_index_t2(self):
        # Testing whether we get the correct index set
        # Input
        instance = "neos-5140963-mincio.mps"
        instance_path = os.path.join(Constants.DATA_MIP, instance)

        # Initialize Gurobi model
        gurobi_venv, GUROBI_FOUND = init_gurobi.initialize_gurobi()
        model = grb.read(instance_path, env=gurobi_venv)
        model.setParam("OutputFlag", 0)
        variables = model.getVars()

        # Set discrete indexes
        discrete = []
        for var in variables:
            if var.VType == grb.GRB.INTEGER or var.VType == grb.GRB.BINARY:
                discrete.append(var.index)

        destroy_size = int(0.05 * len(discrete))

        destroy_set = set(rnd_state.choice(discrete, destroy_size))

        self.assertEqual(destroy_set, {33, 87, 100, 112, 115, 116, 129, 134, 164})

    def test_initial_destroy_t3(self):
        # Testing whether we get the correct index set
        # Input
        instance = "model.lp"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Mutation_25]
        repair_ops = [RepairOperators.Repair]

        # MIP is an instance of _BaseMIP created from given mip instance
        mip = create_mip_solver(instance_path, seed, mip_solver_str="gurobi")
        instance = _Instance(mip)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Initial state and solution
        initial_state = _State(instance, initial_index_to_val, initial_obj_val)

        # Assert
        self.assertEqual(initial_state.destroy_set, None)

    def test_one_iteration_obj_t4(self):

        # Input
        instance = "model.lp"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Mutation_25]
        repair_ops = [RepairOperators.Repair]

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver="gurobi")

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        # Assert
        self.assertEqual(result.best_state.objective(), 4)
