import os
from typing import NamedTuple

import mabwiser.utils


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """

    default_seed = 123456
    """The default random seed."""

    # Optimization sense
    minimize = "minimize"
    maximize = "maximize"

    # Scip variable types
    binary = "BINARY"
    integer = "INTEGER"
    continuous = "CONTINUOUS"

    # Column names for features df
    var_type = "var_type"
    var_lb = "var_lb"
    var_ub = "var_ub"

    # for Big-M constraint
    M = 1000

    # Data folder constants
    _DATA_DIR_NAME = "data"
    _DATA_DIR_MIP_NAME = _DATA_DIR_NAME + os.sep + "miplib"
    _DATA_DIR_MIPGZ_NAME = _DATA_DIR_NAME + os.sep + "miplib_gz"
    _TEST_DIR_NAME = "tests"
    _DATA_DIR_TOY_NAME = _TEST_DIR_NAME + os.sep + "toy"

    # Data paths
    _FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_MIP = _FILE_DIR + os.sep + ".." + os.sep + _DATA_DIR_MIP_NAME
    DATA_MIP_GZ = _FILE_DIR + os.sep + ".." + os.sep + _DATA_DIR_MIPGZ_NAME
    DATA_TOY = _FILE_DIR + os.sep + ".." + os.sep + _DATA_DIR_TOY_NAME

    # Random solution stopping criteria
    random_gap_ub1 = 0.95
    random_gap_ub2 = 0.90

    # theta for proximity destroy heuristic
    theta = 1


def create_rng(seed):
    return mabwiser.utils.create_rng(seed)


def check_false(expression: bool, exception: Exception):
    return mabwiser.utils.check_false(expression, exception)


def check_true(expression: bool, exception: Exception):
    return mabwiser.utils.check_true(expression, exception)
