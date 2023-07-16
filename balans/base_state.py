from balans.base_instance import _Instance
from typing import Any, Dict


class _State:
    """
    State of an instance with solution and objective
    """

    def __init__(self, instance: _Instance, var_to_val: Dict[Any, float], obj_val: float):
        self.instance = instance
        self.var_to_val = var_to_val
        self.obj_val = obj_val

        # Instance variables
        self.destroy_set = None         # dynamic, set by the destroy operator

    def objective(self):
        return self.obj_val

    def solve_and_update(self):

        # Solve the current state with the destroyed variables and update
        self.var_to_val, self.obj_val = self.instance.solve(self.destroy_set, self.var_to_val)