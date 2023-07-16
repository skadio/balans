from typing import NamedTuple, NewType, Union

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

    # Column names for features df
    var_type = "var_type"
    var_lb = "var_lb"
    var_ub = "var_ub"

    # Data folder name
    DATA_DIR = "data"


def create_rng(seed):
    return mabwiser.utils.create_rng(seed)


def check_false(expression: bool, exception: Exception):
    return mabwiser.utils.check_false(expression, exception)


def check_true(expression: bool, exception: Exception):
    return mabwiser.utils.check_true(expression, exception)