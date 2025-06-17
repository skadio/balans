import os

from balans.solver import ParBalans
from balans.utils import Constants
from tests.test_base import BaseTest


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
