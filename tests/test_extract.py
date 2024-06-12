import os
from balans.utils import Constants
from tests.test_base import BaseTest
from balans.base_instance import _Instance


class IndexExtractionTest(BaseTest):

    def test_extract(self):
        # Input
        instance = "test5.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        instance = _Instance(instance_path)
        instance.initial_solve(None)

        print("Discrete index:", instance.discrete_indexes)
        print("Binary index:", instance.binary_indexes)
        print("Integer index:", instance.integer_indexes)

        self.assertEqual(instance.discrete_indexes, [0,1,2,3,4])
        self.assertEqual(instance.binary_indexes, [0,1])
        self.assertEqual(instance.integer_indexes, [2,3,4])

