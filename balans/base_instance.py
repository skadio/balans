import pandas as pd
import pyscipopt as scip
from balans.utils import Constants
from typing import Tuple, Dict, Any
import math
from pyscipopt import Model, quicksum


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
        self.features_df = None  # static, set once and for all in solve()
        self.discrete_indexes = None  # static, set once and for all in solve()
        self.binary_indexes = None  # static, set once and for all in solve()
        self.sense = None  # static, set once and for all in solve()
        self.lp_index_to_val = None  # static, set once and for in solve()
        self.lp_obj_value = None  # static, set once and for in solve()
        self.random_index_to_val = None  # static, second random solution
        self.random_obj_value = None  # static, second random solution

    def solve(self,
              is_initial_solve=False,
              index_to_val=None,
              destroy_set=None,
              dins_binary_set=None,
              rens_float_set=None,
              is_zero_obj=False) -> Tuple[Dict[Any, float], float]:

        print("\t Solve")

        if is_initial_solve:
            return self.initial_solve(index_to_val)
        else:
            # Model
            model = scip.Model()

            # Setting the verbosity level to 0
            model.hideOutput()

            # # Instance
            model.readProblem(self.path)

            # Variables
            variables = model.getVars()

            # DESTROY used for DINS, Local Branching, Mutation, RINS
            if destroy_set:
                for var in variables:
                    # Only consider discrete vars (binary and integer)
                    if var.getIndex() in self.discrete_indexes:
                        # IF not in destroy, fix it
                        if var.getIndex() not in destroy_set:
                            model.addCons(var == index_to_val[var.getIndex()])
                        else:
                            # IF in destroy, and DINS is active, don't fix but add bounding constraint
                            if dins_binary_set:
                                index = var.getIndex()
                                current_lp_diff = abs(index_to_val[index] - self.lp_index_to_val[index])
                                model.addCons(abs(var - self.lp_index_to_val[index]) <= current_lp_diff)

            # DINS: binary variables have more strict condition to be fixed
            if dins_binary_set:
                for var in variables:
                    if var.getIndex() in dins_binary_set:
                        # Fix all binary vars in dins_binary
                        model.addCons(var == index_to_val[var.getIndex()])

            # RENS: Discrete variables, where the lp relaxation is not integral
            if rens_float_set:
                for var in variables:
                    if var.getIndex() in rens_float_set:
                        # Restrict discrete vars to round up and down integer version of it
                        model.addCons(var <= math.floor(index_to_val[var.getIndex()]))
                        model.addCons(var >= math.ceil(index_to_val[var.getIndex()]))

            # Zero Objective
            if is_zero_obj:
                model.setObjective(0, self.sense)

            # Solve
            model.optimize()

            # Solution and objective
            index_to_val = self.get_index_to_val(model)
            obj_value = model.getObjVal()

            print("\t Solve DONE!")
            print("\t index_to_val: ", index_to_val)

            return index_to_val, obj_value

    def extract_features(self, model, variables):

        # Variable types
        var_types = [v.vtype() for v in variables]

        # Set discrete indexes
        discrete = []
        for var in variables:
            if self.is_discrete(var.vtype()):
                discrete.append(var.getIndex())

        self.discrete_indexes = discrete
        print(discrete)

        # Set binary indexes
        binary = []
        for var in variables:
            if self.is_binary(var.vtype()):
                binary.append(var.getIndex())

        self.binary_indexes = binary
        print(binary)

        # Feature df with types and bounds
        # %TODO where is this used at all??
        self.features_df = pd.DataFrame({Constants.var_type: var_types,
                                         Constants.var_lb: [v.getLbGlobal() for v in variables],
                                         Constants.var_ub: [v.getUbGlobal() for v in variables]})

        # Optimization direction
        self.sense = model.getObjectiveSense()

    def initial_solve(self, index_to_val) -> Tuple[Dict[Any, float], float]:

        # Model with verbosity level 0
        model = scip.Model()
        model.hideOutput()

        # Instance
        model.readProblem(self.path)
        # model.setPresolve(scip.SCIP_PARAMSETTING.OFF)

        # Search only for the first incumbent
        model.setParam("limits/bestsol", 1)

        # Variables
        variables = model.getVars()

        # If a solution is given fix it. Can be partial (denoted by None value)
        if index_to_val is not None:
            for var in variables:
                if index_to_val[var.getIndex()] is not None:
                    model.addCons(var == index_to_val[var.getIndex()])

        # Initial solve extracts static instance features
        self.extract_features(model, variables)

        # Solve
        model.optimize()

        # Initial solution and objective
        index_to_val = self.get_index_to_val(model)
        obj_value = model.getObjVal()

        # Reset problem
        model.freeProb()

        # Solve LP relaxation and save it
        self.lp_index_to_val, self.lp_obj_value = self.lp_solve()

        # Create two more random solutions for crossover heuristics** Only needed for Crossover
        self.random_index_to_val, self.random_obj_value = self.get_random_solution(0.80, 20)

        # Return solution
        return index_to_val, obj_value

    def lp_solve(self) -> Tuple[Dict[Any, float], float]:

        # Model with verbosity level 0
        model = scip.Model()
        model.hideOutput()

        # Instance
        model.readProblem(self.path)

        # Variables
        variables = model.getVars()

        # Continuous relaxation of the problem
        for var in variables:
            model.chgVarType(var, Constants.continuous)

        # Solve
        model.optimize()

        # Solution and objective
        index_to_val = self.get_index_to_val(model)
        obj_value = model.getObjVal()

        # Return solution and objective
        return index_to_val, obj_value

    def get_random_solution(self, gap=0.80, time=30) -> Tuple[Dict[Any, float], float]:

        # Model with verbosity level 0
        model = scip.Model()
        model.hideOutput()

        # Instance
        model.readProblem(self.path)
        # model.setPresolve(scip.SCIP_PARAMSETTING.OFF)

        # Search only for the first incumbent
        model.setParam("limits/gap", gap)
        model.setParam("limits/time", time)

        # Variables
        variables = model.getVars()

        # Solve
        model.optimize()

        # Initial solution and objective
        random_index_to_val = self.get_index_to_val(model)
        random_obj_value = model.getObjVal()

        # Reset problem
        model.freeProb()

        # Return solution
        return random_index_to_val, random_obj_value

    @staticmethod
    def is_discrete(var_type) -> bool:
        return var_type in (Constants.binary, Constants.integer)

    @staticmethod
    def is_binary(var_type) -> bool:
        return var_type in Constants.binary

    @staticmethod
    def get_index_to_val(model) -> Dict[Any, float]:
        return dict([(var.getIndex(), model.getVal(var)) for var in model.getVars()])


# #Local Branching V2
# if is_local_branching:
#     currently_zero = []
#     currently_one = []
#     for var in variables:
#         if var.getIndex() in self.binary_indexes:
#             if index_to_val[var.getIndex()] == 0:
#                 currently_zero.append(var.getIndex())
#             else:
#                 currently_one.append(var.getIndex())
#
#     model.addCons(quicksum((var[r] + 1- var[i]) for r in currently_zero for i in currently_one)
#                     <= int(0.5 * len(self.binary_indexes)) )


# if proximity_set:
# TODO -proximity needs modification of constraints.
#     currentNumVar=1
#     for var in variables:
#         # BINARY VARS
#         if var.getIndex() in proximity_set:
#
#             if index_to_val[var.getIndex()] == 0:
#                 model.addVar("var" + str(currentNumVar), vtype="B", obj=1.0)
#
#             if index_to_val[var.getIndex()] == 1:
#                 model.addVar("var" + str(currentNumVar), vtype="B", obj=-1.0)
#
#         currentNumVar = currentNumVar+1
#         model.delVar(var)

# # DROP ALL NON-BINARY VARIABLES
# if var.getIndex() not in proximity_set:
#     model.delVar(var)
