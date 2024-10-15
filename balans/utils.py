import os
from typing import NamedTuple

import mabwiser.utils


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """

    # Default seed
    default_seed = 123456

    # Default MIP Solver
    scip_solver = "scip"
    gurobi_solver = "gurobi"
    default_solver = gurobi_solver

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

    # Time limit for the initial solution to get feasible solution as a starting point for ALNS
    timelimit_first_solution = 3

    # time limit for one iteration is ALNS, local branching has longer time because hard problem created
    timelimit_alns_iteration = 120

    # time limit for one local branching iteration.
    # TODO paper says  Each LNS iteration is limited to 1 minute, except for Local Branching with 2.5 minutes.
    timelimit_local_branching_iteration = 600

    # for Big-M constraint, currently used in Proximity
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


def create_rng(seed):
    return mabwiser.utils.create_rng(seed)


def check_false(expression: bool, exception: Exception):
    return mabwiser.utils.check_false(expression, exception)


def check_true(expression: bool, exception: Exception):
    return mabwiser.utils.check_true(expression, exception)
