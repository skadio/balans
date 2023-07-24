import pandas as pd
import pyscipopt as scip
from balans.utils import Constants
from typing import Tuple, Dict, Any
import math
from pyscipopt import Model


class _Instance:
    """
    Instance from a given MIP file

    instance.solve() operates as a main body of the operators, depending on which operator
    is used its solution procedure changes.

    initial solve > lp solve only for the first iteration.

    non-initial solve > for the rest of the remaining iterations.
    """

    def __init__(self, path):
        self.path = path

        # Instance variables
        self.has_features = False  # Flag to denote if features extracted
        self.features_df = None  # static, set once and for all in solve()
        self.discrete_indexes = None  # static, set once and for all in solve()
        self.binary_indexes = None  # static, set once and for all in solve()
        self.sense = None  # static, set once and for all in solve()
        # self.lp_var_to_val, self.lp_obj = self.lp_solve()

    def solve(self, is_initial_solve=False, destroy_set=None, var_to_val=None, float_index_to_be_bounded=None,
              is_zero_obj=None, dins_set=None, proximity_set=None) -> Tuple[Dict[Any, float], float]:

        # ONLY FOR INITIAL SOLVE
        if is_initial_solve:
            # Model
            model = scip.Model()

            # Setting the verbosity level to 0
            model.hideOutput()

            # # Instance
            model.readProblem(self.path)

            # Turning off presolve
            # model.setPresolve(scip.SCIP_PARAMSETTING.OFF)

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

            model.freeProb()

            # Just solve LP one time together with initial state and use it since it is static.
            self.lp_var_to_val, self.lp_obj_value = self.lp_solve()

            return var_to_val, obj_value, self.lp_var_to_val, self.lp_obj_value

        # IF NOT INITIAL SOLVE:
        if not is_initial_solve:
            # Model
            model = scip.Model()

            # Setting the verbosity level to 0
            model.hideOutput()

            # # Instance
            model.readProblem(self.path)

            # model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
            variables = model.getVars()
            # Features, set once and for all

            if destroy_set:
                for var in variables:
                    if var.getIndex() not in destroy_set:
                        model.addCons(var == var_to_val[var.getIndex()])
                    # ONLY FOR DINS
                    if dins_set:
                        if var.getIndex() in destroy_set:
                            model.addCons(abs(var - self.lp_var_to_val[var.getIndex()]) <= abs(
                                self.lp_var_to_val[var.getIndex()] - var_to_val[var.getIndex()]))

            # ONLY FOR DINS, binary variables have more strict condition, dins_set = binary_indexes
            if dins_set:
                for var in variables:
                    if var.getIndex() in dins_set:
                        model.addCons(var == var_to_val[var.getIndex()])

            # ONLY FOR RENS
            if float_index_to_be_bounded:
                for var in variables:
                    if var.getIndex() in float_index_to_be_bounded:
                        model.addCons(var <= math.floor(var_to_val[var.getIndex()]))
                        model.addCons(var >= math.ceil(var_to_val[var.getIndex()]))

            # ONLY FOR NO OBJECTIVE
            if is_zero_obj:
                model.setObjective(0, self.sense)

            # if proximity_set:
            # TODO -proximity needs modification of constraints.
            #     currentNumVar=1
            #     for var in variables:
            #         # BINARY VARS
            #         if var.getIndex() in proximity_set:
            #
            #             if var_to_val[var.getIndex()] == 0:
            #                 model.addVar("var" + str(currentNumVar), vtype="B", obj=1.0)
            #
            #             if var_to_val[var.getIndex()] == 1:
            #                 model.addVar("var" + str(currentNumVar), vtype="B", obj=-1.0)
            #
            #         currentNumVar = currentNumVar+1
            #         model.delVar(var)

            # # DROP ALL NON-BINARY VARIABLES
            # if var.getIndex() not in proximity_set:
            #     model.delVar(var)

            model.optimize()

            # Solution
            var_to_val = dict([(var.getIndex(), model.getVal(var)) for var in model.getVars()])

            # Objective
            obj_value = model.getObjVal()

            return var_to_val, obj_value

    @staticmethod
    def is_discrete(var_type) -> bool:
        return var_type in (Constants.binary, Constants.integer)

    @staticmethod
    def is_binary(var_type) -> bool:
        return var_type in Constants.binary

    def extract_features(self, model, variables):

        # Set features to true
        self.has_features = True

        # Variable types
        var_types = [v.vtype() for v in variables]

        # Set discrete indexes MODIFIED
        discrete = []
        for var in variables:
            if self.is_discrete(var.vtype()):
                discrete.append(var.getIndex())

        self.discrete_indexes = discrete

        # Set binary indexes MODIFIED
        binary = []
        for var in variables:
            if self.is_binary(var.vtype()):
                binary.append(var.getIndex())

        self.binary_indexes = binary

        # Feature df with types and bounds
        self.features_df = pd.DataFrame({Constants.var_type: var_types,
                                         Constants.var_lb: [v.getLbGlobal() for v in variables],
                                         Constants.var_ub: [v.getUbGlobal() for v in variables]})

        # Optimization direction
        self.sense = model.getObjectiveSense()

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

        # Continuous relaxation of the problem
        for var in variables:
            model.chgVarType(var, 'CONTINUOUS')

        model.optimize()
        # Solution
        var_to_val = dict([(var.getIndex(), model.getVal(var)) for var in model.getVars()])
        # Objective
        obj_value = model.getObjVal()

        return var_to_val, obj_value
