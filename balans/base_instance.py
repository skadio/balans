import math
from typing import Tuple, Dict, Any

from pyscipopt import quicksum

from balans.utils_scip import get_model_and_vars, get_index_to_val_and_objective
from balans.utils_scip import lp_solve, is_binary, is_discrete, split_binary_vars
from balans.utils import Constants


class _Instance:
    """
    Instance from a given MIP file with solve operations on top, subject to operator
    """

    def __init__(self, path):
        # Instance holds the given path
        self.path = path

        # Static, set once and for all
        self.discrete_indexes = None    # all discrete: binary + integer
        self.binary_indexes = None      # discrete binary
        self.integer_indexes = None       # discrete but not binary
        self.sense = None
        self.lp_index_to_val = None
        self.lp_obj_value = None
        self.lp_floating_discrete_indexes = None

    def solve(self,
              is_initial_solve=False,
              index_to_val=None,
              obj_val=None,
              destroy_set=None,
              dins_set=None,
              rens_float_set=None,
              is_zero_obj=False,
              local_branching_size=0,
              is_proximity=False) -> Tuple[Dict[Any, float], float]:

        print("\t Solve")

        if is_initial_solve:
            return self.initial_solve(index_to_val)
        else:
            # flag to identify if any constraint added or objective function changed
            # If flag = True, optimize the problem and get new sol and obj
            # If flag = False, return the current sol and obj, do not optimize
            has_destroy = False
            # Build model and variables
            model, variables = get_model_and_vars(path=self.path)

            # DESTROY used for Crossover, Mutation, RINS
            if destroy_set:
                if len(destroy_set) > 0:
                    has_destroy = True
                    for var in variables:
                        # Only consider discrete vars (binary and integer)
                        if var.getIndex() in self.discrete_indexes:
                            # IF not in destroy, fix it
                            if var.getIndex() not in destroy_set:
                                model.addCons(var == index_to_val[var.getIndex()])

            # DINS: Discrete Variables, where incumbent and lp relaxation have distance more than 0.5
            if dins_set:
                if len(dins_set) > 0:
                    has_destroy = True
                    for var in variables:
                        # Only consider integer variables
                        if var.getIndex() in self.discrete_indexes:
                            if var.getIndex() in self.integer_indexes and var.getIndex() in dins_set:
                                # Add bounding constraint around initial lp solution
                                index = var.getIndex()
                                current_lp_diff = abs(index_to_val[index] - self.lp_index_to_val[index])
                                model.addCons(abs(var - self.lp_index_to_val[index]) <= current_lp_diff)
                            else:
                                # If binary OR not in dins set, fix the var
                                model.addCons(var == index_to_val[var.getIndex()])

            # Local Branching: Binary variables, flip a limited subset (can come from DINS with delta)
            if local_branching_size > 0:
                has_destroy = True
                # Only change a subset of the binary variables, keep others fixed. e.g.,
                zero_binary_vars, one_binary_vars = split_binary_vars(variables, self.binary_indexes, index_to_val)

                # if current binary var is 0, flip to 1 consumes 1 unit of budget
                # if current binary var is 1, flip to 0 consumes 1 unit of budget by (1-x)
                zero_expr = quicksum(zero_var for zero_var in zero_binary_vars)
                one_expr = quicksum(1 - one_var for one_var in one_binary_vars)
                model.addCons(zero_expr + one_expr <= local_branching_size)

            # Proximity: Binary variables, modify objective, add new constraint
            if is_proximity:
                has_destroy = True
                # add cutoff constraint depending on sense, so that next state is better quality
                # a slack variable z to prevent infeasible solution, \theta = 1
                z = model.addVar(vtype=Constants.continuous, lb=0)
                if self.sense == Constants.minimize:
                    model.addCons(model.getObjective() <= obj_val - Constants.theta + z)
                else:
                    model.addCons(model.getObjective() >= obj_val + Constants.theta + z)

                zero_binary_vars, one_binary_vars = split_binary_vars(variables, self.binary_indexes, index_to_val)
                # if x_inc=0, new objective expression is x_inc.
                # if x_inc=1, new objective expression is 1 - x_inc.
                # Drop all other vars (when not in the expr it is set to 0 by default)
                zero_expr = quicksum(zero_var for zero_var in zero_binary_vars)
                one_expr = quicksum(1 - one_var for one_var in one_binary_vars)

                # M * z is to make sure model does not use z, unless needed to avoid infeasibility
                model.setObjective(zero_expr + one_expr + Constants.M * z, Constants.minimize)

            # RENS: Discrete variables, where the lp relaxation is not integral
            if rens_float_set:
                if len(rens_float_set) > 0:
                    has_destroy = True
                    for var in variables:
                        if var.getIndex() in rens_float_set:
                            # Restrict discrete vars to round up and down integer version of it
                            # EX: If var = 3.5, the constraint is var >= 3 and var <= 4
                            model.addCons(var >= math.floor(index_to_val[var.getIndex()]))
                            model.addCons(var <= math.ceil(index_to_val[var.getIndex()]))
                        else:
                            # If not in the set, fix the var
                            model.addCons(var == index_to_val[var.getIndex()])

            # Zero Objective
            if is_zero_obj:
                has_destroy = True
                model.setObjective(0, self.sense)

            # If no destroy, dont's solve, quit with previous objective
            if not has_destroy:
                print("No destroy to apply, don't call optimize()")
                print("\t Current Obj:", obj_val)
                print("\t index_to_val: ", index_to_val)
                return index_to_val, obj_val

            # If destroy, solve for next state
            model.optimize()
            index_to_val, obj_val = get_index_to_val_and_objective(model)

            # Need to find the original obj value for transformed objectives
            if is_proximity or is_zero_obj:

                # Build model and variables
                # This resets the objective back to original
                model, variables = get_model_and_vars(path=self.path)

                # Solution of transformed problem
                var_to_val = model.createSol()
                for i in range(model.getNVars()):
                    var_to_val[variables[i]] = index_to_val[i]

                # Objective value of the solution found in transformed
                print("\t Transformed obj: ", obj_val)
                obj_val = model.getSolObjVal(var_to_val)

            print("\t Solve DONE!", obj_val)
            print("\t index_to_val: ", index_to_val)

            return index_to_val, obj_val

    def initial_solve(self, index_to_val) -> Tuple[Dict[Any, float], float]:

        # Build model and variables
        model, variables = get_model_and_vars(path=self.path, solution_count=1)

        # If a solution is given fix it. Can be partial (denoted by None value)
        if index_to_val is not None:
            for var in variables:
                if index_to_val[var.getIndex()] is not None:
                    model.addCons(var == index_to_val[var.getIndex()])

        # Initial solve extracts static instance features
        self.extract_base_features(model, variables)

        # Solve
        model.optimize()
        index_to_val, obj_value = get_index_to_val_and_objective(model)

        # Reset problem
        model.freeProb()

        self.extract_lp_features(self.path)

        # Return solution
        return index_to_val, obj_value

    def extract_base_features(self, model, variables):

        # Set indexes
        self.discrete_indexes = []
        self.binary_indexes = []
        self.integer_indexes = []

        for var in variables:
            if is_discrete(var.vtype()):
                self.discrete_indexes.append(var.getIndex())
                if is_binary(var.vtype()):
                    self.binary_indexes.append(var.getIndex())
                else:
                    self.integer_indexes.append(var.getIndex())

        # # Set discrete indexes
        # self.discrete_indexes = []
        # for var in variables:
        #     if is_discrete(var.vtype()):
        #         self.discrete_indexes.append(var.getIndex())
        #
        # # Set binary indexes
        # self.binary_indexes = []
        # for var in variables:
        #     if is_binary(var.vtype()):
        #         self.binary_indexes.append(var.getIndex())

        # Optimization direction
        self.sense = model.getObjectiveSense()

    def extract_lp_features(self, path):
        # Solve LP relaxation and save it
        self.lp_index_to_val, self.lp_obj_value = lp_solve(path)

        # list of discrete indexes that are floating point in LP
        self.lp_floating_discrete_indexes = [i for i in self.discrete_indexes if
                                             not self.lp_index_to_val[i].is_integer()]
