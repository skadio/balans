import pyscipopt as scip
from base_state import State
import pandas as pd


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
