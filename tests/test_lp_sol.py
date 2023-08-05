import os
from alns.accept import *
from alns.select import *
from alns.stop import *
from pyscipopt import Model
import numpy as np

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


class LPSolTest(BaseTest):

    def test_lp_t1(self):
        # Input

        # 3 DISCRETE DECISION VARIABLES > model.cip
        instance = "model.cip"
        instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR_TOY, instance)

        instance = _Instance(instance_path)

        # LP solution
        lp_index_to_val, lp_obj_val = instance.lp_solve()

        self.assertEqual(lp_obj_val, 2.666666666666667)
