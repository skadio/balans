import pyscipopt as scip
from balans.utils import Constants
from typing import Tuple, Dict, Any
import math
from pyscipopt import quicksum


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
              dins_random_set=None,
              is_dins=False,
              rens_float_set=None,
              is_zero_obj=False,
              local_branching_size=0,
              is_proximity=False) -> Tuple[Dict[Any, float], float]:

        print("\t Solve")

        if is_initial_solve:
            return self.initial_solve(index_to_val)
        else:
            # Build model and variables
            model, variables = _Instance.get_model_and_vars(path=self.path)

            # DESTROY used for Crossover, DINS, Mutation, RINS
            if destroy_set:
                for var in variables:
                    # Only consider discrete vars (binary and integer)
                    if var.getIndex() in self.discrete_indexes:
                        # IF not in destroy, fix it
                        if var.getIndex() not in destroy_set:
                            # constraint = var == index_to_val[var.getIndex()]
                            # model.addCons(constraint)
                            model.addCons(var == index_to_val[var.getIndex()])
                        else:
                            # IF in destroy, and DINS is active, don't fix but add bounding constraint
                            if is_dins or dins_random_set:
                                index = var.getIndex()
                                current_lp_diff = abs(index_to_val[index] - self.lp_index_to_val[index])
                                model.addCons(abs(var - self.lp_index_to_val[index]) <= current_lp_diff)

            # RANDOM DINS: binary variables have more strict condition to be fixed
            if dins_random_set:
                for var in variables:
                    if var.getIndex() in dins_random_set:
                        # Fix all binary vars in dins_binary
                        model.addCons(var == index_to_val[var.getIndex()])

            # Local Branching: Binary variables, flip a limited subset
            if local_branching_size > 0:
                zero_binary_vars = []
                one_binary_vars = []
                for var in variables:
                    if var.getIndex() in self.binary_indexes:
                        if index_to_val[var.getIndex()] == 0:
                            zero_binary_vars.append(var)
                        else:
                            one_binary_vars.append(var)

                # Only change a subset of the variables, keep others fixed. e.g.,
                # if current binary var is 0, flip to 1 consumes 1 unit of budget
                # if current binary var is 1, flip to 0 consumes 1 unit of budget by (1-x)
                expr = quicksum(zero_var for zero_var in zero_binary_vars) + quicksum(1 - one_var for one_var in one_binary_vars)
                model.addCons(expr <= local_branching_size)

            # Proximity: Binary variables, flip their objective
            if is_proximity:
                zero_binary_vars = []
                one_binary_vars = []
                for var in variables:
                    if var.getIndex() in self.binary_indexes:
                        if index_to_val[var.getIndex()] == 0:
                            zero_binary_vars.append(var)
                        else:
                            one_binary_vars.append(var)

                # if x_inc=0, update its objective coefficient to 1.
                # if x_inc=1, update its objective coefficient to -1.
                # Drop all other vars (when not in the expr it is set to 0 by default)
                zero_obj = quicksum(zero_var for zero_var in zero_binary_vars)
                one_obj = quicksum(-1 * one_var for one_var in one_binary_vars)
                model.setObjective(zero_obj + one_obj, self.sense)

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
            index_to_val = self.get_index_to_val(model)
            obj_value = model.getObjVal()

            # Update Objective for transformed objectives
            if is_proximity or is_zero_obj:

                # Build model and variables
                # TODO why build again?
                model, variables = _Instance.get_model_and_vars(path=self.path)

                # Solution of transformed problem
                # TODO do we have unit test to test this solution switch
                var_to_val = model.createSol()
                for i in range(model.getNVars()):
                    var_to_val[variables[i]] = index_to_val[i]

                # Update objective value
                obj_value = model.getSolObjVal(var_to_val)

            print("\t Solve DONE!")
            print("\t index_to_val: ", index_to_val)

            return index_to_val, obj_value

    def extract_features(self, model, variables):

        # Set discrete indexes
        self.discrete_indexes = []
        for var in variables:
            if self.is_discrete(var.vtype()):
                self.discrete_indexes.append(var.getIndex())

        # Set binary indexes
        self.binary_indexes = []
        for var in variables:
            if self.is_binary(var.vtype()):
                self.binary_indexes.append(var.getIndex())

        # Optimization direction
        self.sense = model.getObjectiveSense()

    def initial_solve(self, index_to_val) -> Tuple[Dict[Any, float], float]:

        # Build model and variables
        model, variables = _Instance.get_model_and_vars(path=self.path, solution_count=1)

        # If a solution is given fix it. Can be partial (denoted by None value)
        if index_to_val is not None:
            for var in variables:
                if index_to_val[var.getIndex()] is not None:
                    model.addCons(var == index_to_val[var.getIndex()])

        # Initial solve extracts static instance features
        self.extract_features(model, variables)

        # Solve
        model.optimize()
        index_to_val = self.get_index_to_val(model)
        obj_value = model.getObjVal()

        # Reset problem
        # TODO Why is this needed?
        model.freeProb()

        # Solve LP relaxation and save it
        self.lp_index_to_val, self.lp_obj_value = self.lp_solve()

        # Create two more random solutions for crossover heuristics** Only needed for Crossover
        self.random_index_to_val, self.random_obj_value = self.random_solve(0.80, 20)

        # Return solution
        return index_to_val, obj_value

    def lp_solve(self) -> Tuple[Dict[Any, float], float]:

        # Build model and variables
        model, variables = _Instance.get_model_and_vars(path=self.path,
                                                        is_lp_relaxation=True)

        # Solve
        model.optimize()
        index_to_val = self.get_index_to_val(model)
        obj_value = model.getObjVal()

        # Return solution and objective
        return index_to_val, obj_value

    def random_solve(self, gap=Constants.random_gap, time=Constants.random_time) -> Tuple[Dict[Any, float], float]:

        # Build model and variables
        model, variables = _Instance.get_model_and_vars(path=self.path, gap=gap, time=time)

        # Solve
        model.optimize()
        random_index_to_val = self.get_index_to_val(model)
        random_obj_value = model.getObjVal()

        # Reset problem
        # TODO Why is this needed?
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

    @staticmethod
    def get_model_and_vars(path, is_verbose=False, has_pre_solve=True,
                           solution_count=None, gap=None, time=None,
                           is_lp_relaxation=False):

        # Model
        model = scip.Model()

        # Verbosity
        if not is_verbose:
            model.hideOutput()

        # Instance
        model.readProblem(path)

        if not has_pre_solve:
            model.setPresolve(scip.SCIP_PARAMSETTING.OFF)

        # Search only for the first incumbent
        if solution_count == 1:
            model.setParam("limits/bestsol", 1)

        # Search only for the first incumbent
        if gap is not None:
            model.setParam("limits/gap", gap)

        if time is not None:
            model.setParam("limits/time", time)

        # Variables
        variables = model.getVars()

        # Continuous relaxation of the problem
        if is_lp_relaxation:
            for var in variables:
                model.chgVarType(var, Constants.continuous)

        # Return model and vars
        return model, variables
