from balans.base_instance import _Instance
from typing import Any, Dict


class _State:
    """
    State of an instance with solution and objective
    """

    def __init__(self, instance: _Instance, var_to_val: Dict[Any, float], obj_val: float,destroy_set =None,float_index_to_be_bounded=None):
        self.instance = instance
        self.var_to_val = var_to_val
        self.obj_val = obj_val
        self.float_index_to_be_bounded=float_index_to_be_bounded

        # Instance variables
        self.destroy_set = destroy_set         # dynamic, set by the destroy operator

    def objective(self):
        return self.obj_val

    def solve_and_update(self):

        # Solve the current state with the destroyed variables and update
        self.var_to_val, self.obj_val = self.instance.solve(destroy_set=self.destroy_set,var_to_val= self.var_to_val,float_index_to_be_bounded=self.float_index_to_be_bounded)