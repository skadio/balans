from balans.base_instance import _Instance
from typing import Any, Dict


class _State:
    """
    State of an instance with solution and objective
    """

    def __init__(self,
                 instance: _Instance,
                 index_to_val: Dict[Any, float],
                 obj_val: float,
                 destroy_set=None,
                 dins_binary_set=None,
                 proximity_set=None,
                 rens_float_set=None,
                 is_zero_obj=None):

        self.instance = instance

        # Var index is defined by SCIP as var.getIndex()
        self.index_to_val = index_to_val
        self.obj_val = obj_val

        # Instance variables
        self.destroy_set = destroy_set
        self.dins_binary_set = dins_binary_set
        self.proximity_set = proximity_set
        self.rens_float_set = rens_float_set
        self.is_zero_obj = is_zero_obj

    def solution(self):
        return self.index_to_val

    def objective(self):
        return self.obj_val

    def reset_solve_settings(self):
        self.destroy_set = None
        self.dins_binary_set = None
        self.proximity_set = None
        self.rens_float_set = None
        self.is_zero_obj = None

    def solve_and_update(self):
        # Solve the current state with the destroyed variables and update
        self.index_to_val, self.obj_val = self.instance.solve(is_initial_solve=False,
                                                              index_to_val=self.index_to_val,
                                                              destroy_set=self.destroy_set,
                                                              dins_binary_set=self.dins_binary_set,
                                                              proximity_set=self.proximity_set,
                                                              rens_float_set=self.rens_float_set,
                                                              is_zero_obj=self.is_zero_obj)
