import os

from balans.base_instance import _Instance
from balans.base_mip import create_mip_solver
from balans.utils import Constants
from tests.test_base import BaseTest


class IndexExtractionTest(BaseTest):

    def test_extract(self):
        # Input
        instance = "test5.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # MIP is an instance of _BaseMIP created from given mip instance
        mip = create_mip_solver(instance_path)
        instance = _Instance(mip)

        instance.initial_solve(None)

        print("Discrete index:", instance.discrete_indexes)
        print("Binary index:", instance.binary_indexes)
        print("Integer index:", instance.integer_indexes)

        self.assertEqual(instance.discrete_indexes, [0, 1, 2, 3, 4])
        self.assertEqual(instance.binary_indexes, [0, 1])
        self.assertEqual(instance.integer_indexes, [2, 3, 4])

