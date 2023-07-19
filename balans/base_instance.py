import pandas as pd
import pyscipopt as scip
from balans.utils import Constants
from typing import Tuple, Dict, Any
import math
from pyscipopt import Model


class _Instance:
    """
    Instance from a given MIP file
    """

    def __init__(self, path):
        self.path = path

        # Instance variables
        self.has_features = False  # Flag to denote if features extracted
        self.features_df = None  # static, set once and for all in solve()
        self.discrete_indexes = None  # static, set once and for all in solve()
        self.binary_indexes = None  # static, set once and for all in solve()
        self.sense = None  # static, set once and for all in solve()

    def solve(self, is_initial_solve=False, destroy_set=None, var_to_val=None,float_index_to_be_bounded=None) -> Tuple[Dict[Any, float], float]:

        if is_initial_solve:
            # Model
            model = scip.Model()
            model.hideOutput()
            model.readProblem(self.path)

            #
            # # Instance
            model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
            model.setParam("limits/bestsol", 1)
            variables = model.getVars()
            # Features, set once and for all
            if not self.has_features:
                self.extract_features(model, variables)
            model.optimize()
            # Solution
            var_to_val = dict([(var.getIndex(), model.getVal(var)) for var in model.getVars()])
            # Objective
            obj_value = model.getObjVal()

            if is_initial_solve:
                model.freeProb()
            return var_to_val, obj_value
        if not is_initial_solve:
            # Model
            model = scip.Model()
            model.hideOutput()
            model.readProblem(self.path)
            #
            # # Instance
            # model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
            # model.setParam("limits/bestsol", 100)
            variables = model.getVars()
            # Features, set once and for all
            if not self.has_features:
                self.extract_features(model, variables)

            if destroy_set:
                for var in variables:
                    if var.getIndex() not in destroy_set:
                        model.addCons(var == var_to_val[var.getIndex()])

            if float_index_to_be_bounded:
                for var in variables:
                    if var.getIndex() in float_index_to_be_bounded:
                        model.addCons(var <= math.floor(var_to_val[var.getIndex()]))
                        model.addCons(var >= math.ceil(var_to_val[var.getIndex()]))

            model.optimize()

            # Solution
            var_to_val = dict([(var.getIndex(), model.getVal(var)) for var in model.getVars()])

            # Objective
            obj_value = model.getObjVal()

            return var_to_val, obj_value

    @staticmethod
    def is_discrete(var_type) -> bool:
        return var_type in (Constants.binary, Constants.integer)

    def extract_features(self, model, variables):

        # Set features to true
        self.has_features = True

        # Variable types
        var_types = [v.vtype() for v in variables]

        # Set discrete indexes MODIFIED
        discrete = []
        for var in variables:
            if var.vtype() == 'INTEGER' or var.vtype() == 'BINARY':
                discrete.append(var.getIndex())

        self.discrete_indexes = discrete

        # Set binary indexes MODIFIED
        binary = []
        for var in variables:
            if var.vtype() == 'BINARY':
                binary.append(var.getIndex())

        self.binary_indexes = binary

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

    def lp_solve(self, destroy_set=None, var_to_val=None) -> Tuple[Dict[Any, float], float]:

        # Model
        model = scip.Model()
        model.hideOutput()

        # Instance
        model.readProblem(self.path)

        variables = model.getVars()
        # Features, set once and for all
        if not self.has_features:
            self.extract_features(model, variables)

        for var in variables:
            # Continuous relaxation of the problem
            model.chgVarType(var, 'CONTINUOUS')

        model.optimize()
        # Solution
        var_to_val = dict([(var.getIndex(), model.getVal(var)) for var in model.getVars()])
        # Objective
        obj_value = model.getObjVal()

        return var_to_val, obj_value
