import os

import numpy as np
from alns.ALNS import ALNS
from alns.accept import *
from alns.select import *
from alns.stop import *
from mabwiser.mab import LearningPolicy

from balans.base_instance import _Instance
from balans.base_mip import create_mip_solver
from balans.base_state import _State
from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants
from tests.test_base import BaseTest
from balans.solver import ParBalans

class ParBalansTest(BaseTest):

    BaseTest.mip_solver = Constants.gurobi_solver

    def test_parbalans(self):
        # Input
        instance = "noswot.mps"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        balans = ParBalans(n_jobs=2)
        result = balans.run(instance_path)

        print("Best solution:", result[0])

        # self.assertEqual(result.best_state.objective(), 4)