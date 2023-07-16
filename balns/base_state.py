import pandas as pd
import pyscipopt as scip
from balns.utils import Constants
import typing


class State:
    """
    State of an instance with solution and objective
    """

    def __init__(self, instance, var_to_val, obj_val):
        self.instance = instance
        self.var_to_val = var_to_val
        self.obj_val = obj_val

        # Instance variables
        self.destroy_set = None         # dynamic, set by the destroy operator

    def objective(self):
        return self.obj_val

    def solve_and_update(self, gap=None, time=None):

        # Solve the current state with the destroyed variables and update
        self.var_to_val, self.obj_val = self.instance.solve(gap, time, self.destroy_set, self.var_to_val)

    @staticmethod
    def is_discrete(var_type):
        return var_type in (Constants.Binary, Constants.Integer)

    def extract_features(self, variables):

        # Set features to true
        self.has_features = True

        # Variable types
        var_types = [v.vtype() for v in variables]

        # Set discrete indexes
        self.discrete_indexes = [i for i, var_type in enumerate(var_types) if self.is_discrete(var_type)]

        # Feature df with types and bounds
        self.features_df = pd.DataFrame({Constants.var_type: var_types,
                                         Constants.var_lb: [v.getLbGlobal() for v in variables],
                                         Constants.var_ub: [v.getUbGlobal() for v in variables]})

        # Change df types
        self.features_df = self.features_df.astype({Constants.var_type: int,
                                                    Constants.var_lb: float,
                                                    Constants.var_ub: float})

        # Other possible features can be LP relaxation?

