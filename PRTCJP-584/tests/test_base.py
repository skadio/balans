# -*- coding: utf-8 -*-

import unittest
from typing import List, Union, Optional


class BaseTest(unittest.TestCase):

    # A list common to all tests 
    xx = []

    @staticmethod
    def common_test_function(some_parameter) -> Union[List[float], List[List[float]]]:
        """Sets up a common test

        Return 
        """
        return -1

    def assertListAlmostEqual(self, list1, list2):
        """
        Asserts that floating values in the given lists (almost) equals to each other
        """
        if not isinstance(list1, list):
            list1 = list(list1)

        if not isinstance(list2, list):
            list2 = list(list2)

        self.assertEqual(len(list1), len(list2))

        for index, val in enumerate(list1):
            self.assertAlmostEqual(val, list2[index])
