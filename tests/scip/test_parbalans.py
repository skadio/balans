import os
import pickle

from balans.solver import ParBalans
from balans.utils import Constants
from tests.test_base import BaseTest


class ParBalansTest(BaseTest):
    BaseTest.mip_solver = Constants.scip_solver

    def test_parbalans(self):
        # Input
        instance = "noswot.mps"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        parbalans = ParBalans(n_jobs=2,
                              n_mip_jobs=1,
                              mip_solver=BaseTest.mip_solver,
                              output_dir="results/",
                              balans_generator=ParBalans._generate_random_balans)
        result = parbalans.run(instance_path)

        print("Best solution:", result[0])
        self.assertEqual(result[1], -41)

        with open("results/result_0.pkl", "rb") as file:
            result0 = pickle.load(file)
        with open("results/result_1.pkl", "rb") as file:
            result1 = pickle.load(file)
        if result0[0][-1] < result1[0][-1]:
            self.assertEqual(result0[0][-1], -41)
        else:
            self.assertEqual(result1[0][-1], -41)