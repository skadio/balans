from balans.base_instance import _Instance
from typing import Any, Dict


class _State:
    """
    State of an instance with solution and objective
    """

    def __init__(self,
                 instance: _Instance,
                 var_to_val: Dict[Any, float],
                 obj_val: float,
                 destroy_set=None,
                 dins_binary_set=None,
                 rens_float_set=None,
                 is_zero_obj=None,
                 proximity_set=None):

        self.instance = instance

        # Var is not a scip object but variable index returned by scip via var.getIndex()
        self.var_to_val = var_to_val
        self.obj_val = obj_val

        # Instance variables
        self.destroy_set = destroy_set
        self.dins_set = dins_binary_set
        self.rens_float_set = rens_float_set
        self.is_zero_obj = is_zero_obj
        self.proximity_set = proximity_set

    def objective(self):
        return self.obj_val

    def reset_solve_settings(self):
        self.destroy_set = None
        self.dins_set = None
        self.rens_float_set = None
        self.is_zero_obj = None
        self.proximity_set = None

    def solve_and_update(self):
        # Solve the current state with the destroyed variables and update
        self.var_to_val, self.obj_val = self.instance.solve(destroy_set=self.destroy_set,
                                                            var_to_val=self.var_to_val,
                                                            rens_float_set=self.rens_float_set,
                                                            is_zero_obj=self.is_zero_obj, dins_binary_set=self.dins_set,
                                                            proximity_set=self.proximity_set)
