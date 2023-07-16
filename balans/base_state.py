import pandas as pd
import pyscipopt as scip
from balans.utils import Constants
from balans.base_instance import Instance
from typing import Any, Dict


class State:
    """
    State of an instance with solution and objective
    """

    def __init__(self, instance: Instance, var_to_val: Dict[Any, float], obj_val: float):
        self.instance = instance
        self.var_to_val = var_to_val
        self.obj_val = obj_val

        # Instance variables
        self.destroy_set = None         # dynamic, set by the destroy operator

    def objective(self):
        return self.obj_val

    def solve_and_update(self, gap: float = None, time: float = None):

        # Solve the current state with the destroyed variables and update
        self.var_to_val, self.obj_val = self.instance.solve(gap, time, self.destroy_set, self.var_to_val)