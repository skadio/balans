from typing import Tuple, Dict, Any
import math
from pyscipopt import quicksum
from balans.utils_scip import get_model_and_vars, random_solve, get_index_to_val
from balans.utils_scip import lp_solve, is_binary, is_discrete


class _Instance:
    """
    Instance from a given MIP file with solve operations on top, subject to operator
    """

    def __init__(self, path):
        self.path = path

        # Static, set once and for all
        self.discrete_indexes = None
        self.binary_indexes = None
        self.sense = None
        self.lp_index_to_val = None
        self.lp_obj_value = None

        # TODO how come these are static?
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
            model, variables = get_model_and_vars(path=self.path)

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
            index_to_val = get_index_to_val(model)
            obj_value = model.getObjVal()

            # Update Objective for transformed objectives
            if is_proximity or is_zero_obj:

                # Build model and variables
                # TODO why build again?
                model, variables = get_model_and_vars(path=self.path)

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
        index_to_val = get_index_to_val(model)
        obj_value = model.getObjVal()

        # Reset problem
        # TODO Why is this needed?
        model.freeProb()

        self.extract_lp_features(self.path)

        # Create two more random solutions for crossover heuristics**
        # Only needed for Crossover
        # TODO Not sure about creating random with fixed gap (same bnd tree!)
        self.random_index_to_val, self.random_obj_value = random_solve(path=self.path, gap=0.80, time=20)

        # Return solution
        return index_to_val, obj_value

    def extract_base_features(self, model, variables):

        # Set discrete indexes
        self.discrete_indexes = []
        for var in variables:
            if is_discrete(var.vtype()):
                self.discrete_indexes.append(var.getIndex())

        # Set binary indexes
        self.binary_indexes = []
        for var in variables:
            if is_binary(var.vtype()):
                self.binary_indexes.append(var.getIndex())

        # Optimization direction
        self.sense = model.getObjectiveSense()

    def extract_lp_features(self, path):
        # Solve LP relaxation and save it
        self.lp_index_to_val, self.lp_obj_value = lp_solve(path)


