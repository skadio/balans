import os

from balans.base_mip import create_mip_solver
from balans.utils import Constants
from tests.test_base import BaseTest


class LPSolTest(BaseTest):

    def test_lp_t1(self):
        # Input
        # 3 DISCRETE DECISION VARIABLES > model.cip
        instance = "model.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        mip = create_mip_solver(instance_path, 123)

        # LP solution
        lp_index_to_val, lp_obj_val = mip.solve_lp_and_undo()

        self.assertAlmostEqual(lp_obj_val, 2.666666666666667)

    def test_lp_t2(self):
        # Input

        instance = "gen-ip054.mps"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        mip = create_mip_solver(instance_path, 123)

        # LP solution
        lp_index_to_val, lp_obj_val = mip.solve_lp_and_undo()

        self.assertAlmostEqual(lp_obj_val, 6765.209042593413)

    def test_lp_t3(self):
        # Input
        instance = "pk1.mps"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        mip = create_mip_solver(instance_path, 123)

        # LP solution
        lp_index_to_val, lp_obj_val = mip.solve_lp_and_undo()

        self.assertAlmostEqual(lp_obj_val, 0.0)
