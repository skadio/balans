import unittest

from balans.utils import Constants


class BaseTest(unittest.TestCase):

    mip_solver = Constants.default_solver

    def assertIsBetter(self, before, after, sense):
        if sense == Constants.minimize:
            self.assertLess(after, before)
        else:
            self.assertGreater(after, before)

    def is_not_worse(self, before, after, sense):
        if sense == Constants.minimize:
            self.assertLessEqual(after, before)
        else:
            self.assertGreaterEqual(after, before)
