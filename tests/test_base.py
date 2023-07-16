import unittest

from balns.utils import Constants


class BaseTest(unittest.TestCase):

    @staticmethod
    def is_better(before, after, sense):
        if sense == Constants.minimize:
            return after < before
        else:
            return after > before

    @staticmethod
    def is_not_worse(before, after, sense):
        if sense == Constants.minimize:
            return after <= before
        else:
            return after >= before