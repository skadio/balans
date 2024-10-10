import os

from balans.base_mip import create_mip_solver
from balans.utils import Constants
from tests.test_base import BaseTest


class SCIPLPSolTest(BaseTest):

    def test_lp_t1(self):
        # Input

        # 3 DISCRETE DECISION VARIABLES > model.cip
        instance = "model.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        mip = create_mip_solver(instance_path, 123, Constants.scip_solver)

        # LP solution
        lp_index_to_val, lp_obj_val = mip.solve_lp_and_undo()

        self.assertAlmostEqual(lp_obj_val, 2.666666666666667)

    def test_lp_t2(self):
        # Input

        # 3 DISCRETE DECISION VARIABLES > model.cip
        instance = "10teams.mps"
        instance_path = os.path.join(Constants.DATA_MIP, instance)

        mip = create_mip_solver(instance_path, 123, Constants.scip_solver)

        # LP solution
        lp_index_to_val, lp_obj_val = mip.solve_lp_and_undo()

        self.assertAlmostEqual(lp_obj_val, 917)

    def test_lp_t3(self):
        # Input

        # 3 DISCRETE DECISION VARIABLES > model.cip
        instance = "30n20b8.mps"
        instance_path = os.path.join(Constants.DATA_MIP, instance)

        mip = create_mip_solver(instance_path, 123, Constants.scip_solver)

        # LP solution
        lp_index_to_val, lp_obj_val = mip.solve_lp_and_undo()

        self.assertAlmostEqual(lp_obj_val, 1.5664076455877098)
