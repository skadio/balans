import os

from balans.solver import ParBalans
from balans.utils import Constants
from tests.test_base import BaseTest


class ParBalansTest(BaseTest):
    BaseTest.mip_solver = Constants.scip_solver

    def test_parbalans(self):
        # Input
        instance = "noswot.mps"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # TODO explicitly set all the arguments to ParBalans
        parbalans = ParBalans(n_jobs=2)
        result = parbalans.run(instance_path)

        # TODO assert the output
        print("Best solution:", result[0])
        print("Best solution:", result[1])

        # TODO read back the results files and assert them too
        # self.assertEqual(result.best_state.objective(), 4)
