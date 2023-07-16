import pandas as pd
import pyscipopt as scip
from balans.utils import Constants
from typing import Tuple, Dict, Any


class Instance:
    """
    Instance from a given MIP file
    """

    def __init__(self, path):
        self.path = path

        # Instance variables
        self.has_features = False  # Flag to denote if features extracted
        self.features_df = None  # static, set once and for all in solve()
        self.discrete_indexes = None  # static, set once and for all in solve()
        self.sense = None  # static, set once and for all in solve()

    def solve(self, gap=None, time=None, destroy_set=None, var_to_val=None) -> Tuple[Dict[Any, float], float]:

        # Model
        model = scip.Model()
        model.hideOutput()

        # Instance
        model.readProblem(self.path)

        # Parameters
        # model.setPresolve(scip.SCIP_PARAMSETTING.OFF) ## we should use presolve, no?
        if gap:
            model.setParam("limits/gap", gap)
        if time:
            model.setParam('limits/time', time)

        # Variables
        variables = model.getVars()

        # Features, set once and for all
        if not self.has_features:
            self.extract_features(model, variables)

        # Fix non-destroy variables
        if destroy_set:
            for var in variables:
                index = var.getIndex()
                if index not in destroy_set:
                    model.addCons(var == var_to_val[index])

        # Solve, potentially with fixed variables
        model.optimize()

        # Solution
        var_to_val = dict([(var.getIndex(), model.getVal(var)) for var in model.getVars()])

        # Objective
        obj_value = model.getObjVal()

        # Return solution and objective
        return var_to_val, obj_value

    @staticmethod
    def is_discrete(var_type) -> bool:
        return var_type in (Constants.binary, Constants.integer)

    def extract_features(self, model, variables):

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

        # # Change df types
        # self.features_df = self.features_df.astype({Constants.var_type: int,
        #                                             Constants.var_lb: float,
        #                                             Constants.var_ub: float})

        # Optimization direction
        self.sense = model.getObjectiveSense()

        # Other possible features can be LP relaxation?
