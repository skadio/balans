import os
from balans.utils import Constants
from tests.test_base import BaseTest
from balans.base_instance import _Instance
from balans.utils_scip import lp_solve

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = TEST_DIR + os.sep + ".." + os.sep


class LPSolTest(BaseTest):

    def test_lp_t1(self):
        # Input

        # 3 DISCRETE DECISION VARIABLES > model.cip
        instance = "model.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR_TOY, instance)

        # LP solution
        lp_index_to_val, lp_obj_val = lp_solve(instance_path)

        self.assertAlmostEquals(lp_obj_val, 2.666666666666667)
