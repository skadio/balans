import math
import random
from typing import Tuple, Dict, Any

from pyscipopt import quicksum, Expr
import pyscipopt as scip

from balans.utils_scip import get_index_to_val_and_objective, lp_solve
from balans.utils_scip import is_binary, is_discrete, split_binary_vars
from balans.utils import Constants


class _Instance:
    """
    Instance from a given MIP file with solve operations on top, subject to operator
    """

    def __init__(self, model, seed=Constants.default_seed):
        # SCIP model
        self.model = model
        self.seed = seed

        # Static, set once and for all
        self.discrete_indexes = None    # all discrete: binary + integer
        self.binary_indexes = None      # discrete binary
        self.integer_indexes = None       # discrete but not binary
        self.sense = None
        self.lp_index_to_val = None
        self.lp_obj_val = None
        self.lp_floating_discrete_indexes = None

    def solve(self,
              index_to_val=None,
              obj_val=None,
              destroy_set=None,
              dins_set=None,
              rens_float_set=None,
              has_random_obj=False,
              local_branching_size=0,
              is_proximity=False) -> Tuple[Dict[Any, float], float]:

        print("\t Solve")
        # has_destroy to identify if any constraint added or objective function changed
        # If has_destroy = True, optimize the problem and get new sol and obj
        # If has_destroy = False, return the current sol and obj, do not optimize
        has_destroy = False
        # Build model and variables
        variables = self.model.getVars()
        org_objective = self.model.getObjective()
        org_index_to_val = index_to_val
        org_obj_val = obj_val
        # Record the constraints we added to the model
        constraints = []

        # DESTROY used for Crossover, Mutation, RINS
        if destroy_set:
            if len(destroy_set) > 0:
                has_destroy = True
                for var in variables:
                    # IF not in destroy, fix it
                    if var.getIndex() not in destroy_set:
                        # fix the variable
                        constraints.append(self.model.addCons(var == index_to_val[var.getIndex()]))

        # DINS: Discrete Variables, where incumbent and lp relaxation have distance more than 0.5
        if dins_set:
            if len(dins_set) > 0:
                has_destroy = True
                for var in variables:
                    if var.getIndex() in dins_set:
                        # Add bounding constraint around initial lp solution
                        index = var.getIndex()
                        current_lp_diff = abs(index_to_val[index] - self.lp_index_to_val[index])
                        constraints.append(self.model.addCons(abs(var - self.lp_index_to_val[index]) <= current_lp_diff))
                    else:
                        # fix the variable
                        constraints.append(self.model.addCons(var == index_to_val[var.getIndex()]))

        # Local Branching: Binary variables, flip a limited subset (can come from DINS with delta)
        if local_branching_size > 0:
            has_destroy = True
            # Only change a subset of the binary variables, keep others fixed. e.g.,
            zero_binary_vars, one_binary_vars = split_binary_vars(variables, self.binary_indexes, index_to_val)

            # if current binary var is 0, flip to 1 consumes 1 unit of budget
            # if current binary var is 1, flip to 0 consumes 1 unit of budget by (1-x)
            zero_expr = quicksum(zero_var for zero_var in zero_binary_vars)
            one_expr = quicksum(1 - one_var for one_var in one_binary_vars)
            constraints.append(self.model.addCons(zero_expr + one_expr <= local_branching_size))

        # Proximity: Binary variables, modify objective, add new constraint
        if is_proximity:
            has_destroy = True
            zero_binary_vars, one_binary_vars = split_binary_vars(variables, self.binary_indexes, index_to_val)
            # if x_inc=0, new objective expression is x_inc.
            # if x_inc=1, new objective expression is 1 - x_inc.
            # Drop all other vars (when not in the expr it is set to 0 by default)
            zero_expr = quicksum(zero_var for zero_var in zero_binary_vars)
            one_expr = quicksum(1 - one_var for one_var in one_binary_vars)

            # add cutoff constraint depending on sense, so that next state is better quality
            # a slack variable z to prevent infeasible solution, \theta = 1
            z = self.model.addVar(vtype=Constants.continuous, lb=0)
            constraints.append(self.model.addCons(self.model.getObjective() <= obj_val * (1 - is_proximity) + z))
            # M * z is to make sure model does not use z, unless needed to avoid infeasibility
            self.model.setObjective(zero_expr + one_expr + Constants.M * z, Constants.minimize)

        # RENS: Discrete variables, where the lp relaxation is not integral
        if rens_float_set:
            if len(rens_float_set) > 0:
                has_destroy = True
                for var in variables:
                    if var.getIndex() in rens_float_set:
                        # Restrict discrete vars to round up and down integer version of the lp
                        # EX: If var = 3.5, the constraint is var >= 3 and var <= 4
                        constraints.append(self.model.addCons(var >= math.floor(self.lp_index_to_val[var.getIndex()])))
                        constraints.append(self.model.addCons(var <= math.ceil(self.lp_index_to_val[var.getIndex()])))
                    else:
                        # If not in the set, fix the var to the current state
                        constraints.append(self.model.addCons(var == index_to_val[var.getIndex()]))

        # Random Objective
        if has_random_obj:
            has_destroy = True
            objective = Expr()
            for var in variables:
                coeff = random.uniform(-1,1)
                if coeff != 0:
                    objective += coeff * var
            objective.normalize()
            self.model.setObjective(objective, self.sense)
            self.model.setParam("limits/bestsol", 1)
            self.model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

        # If no destroy, don't solve, quit with previous objective
        if not has_destroy:
            print("No destroy to apply, don't call optimize()")
            print("\t Current Obj:", obj_val)
            # print("\t index_to_val: ", index_to_val)
            return index_to_val, obj_val

        # TODO add to constants
        if local_branching_size > 0:
            self.model.setParam("limits/time", 600)
        else:
            self.model.setParam("limits/time", 120)

        # destroy, solve for next state
        self.model.optimize()

        index_to_val, obj_val = get_index_to_val_and_objective(self.model)

        # TODO add undo_solve() method
        # Get back the original model
        self.model.freeTransform()
        self.model.setParam("limits/bestsol", -1)
        self.model.setHeuristics(scip.SCIP_PARAMSETTING.DEFAULT)
        for ct in constraints:
            self.model.delCons(ct)
        self.model.setObjective(org_objective, self.sense)
        if is_proximity:
            self.model.delVar(z)

        if len(index_to_val) == 0:
            print("Go back to previous state")
            print("\t Current Obj:", org_obj_val)
            return org_index_to_val, org_obj_val

        # Need to find the original obj value for transformed objectives
        if is_proximity or has_random_obj:
            # Objective value of the solution found in transformed
            print("\t Transformed obj: ", obj_val)

            obj_val = 0
            for key, item in org_objective.terms.items():
                obj_val += item * index_to_val[key[0].getIndex()]

        print("\t Solve DONE!", obj_val)
        return index_to_val, obj_val

    def initial_solve(self, index_to_val) -> Tuple[Dict[Any, float], float]:
        variables = self.model.getVars()

        # Initial solve extracts static instance features
        self.extract_base_features(self.model, variables)
        self.extract_lp_features(self.model)

        # TODO: Revert before existing
        if self.sense == Constants.maximize:
            self.model.setObjective(-self.model.getObjective())
            self.sense = Constants.minimize

        # Record the constraints we added to the model
        constraints = []
        # If a solution is given fix it. Can be partial (denoted by None value)
        if index_to_val is not None:
            for var in variables:
                if index_to_val[var.getIndex()] is not None:
                    constraints.append(self.model.addCons(var == index_to_val[var.getIndex()]))

        # TODO: move constants to utils.Constants
        self.model.setParam("limits/time", 20)
        # Solve to get initial solution
        self.model.optimize()
        index_to_val, obj_val = get_index_to_val_and_objective(self.model)

        # Get back the original model
        self.model.freeTransform()
        self.model.setParam("limits/time", 1e+20)
        for ct in constraints:
            self.model.delCons(ct)

        # If no feasible initial solution found within time limit
        # then, solve for first feasible solution without time limit
        if len(index_to_val) == 0:
            self.model.setParam("limits/bestsol", 1)
            # Solve to get initial solution
            self.model.optimize()
            index_to_val, obj_val = get_index_to_val_and_objective(self.model)

            # Get back the original model
            self.model.freeTransform()
            self.model.setParam("limits/bestsol", -1)

        # Return solution
        return index_to_val, obj_val

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

        # Optimization direction
        self.sense = model.getObjectiveSense()

    def extract_lp_features(self, model):
        self.lp_index_to_val, self.lp_obj_val = lp_solve(model)
        self.lp_floating_discrete_indexes = [i for i in self.discrete_indexes if
                                             not self.lp_index_to_val[i].is_integer()]