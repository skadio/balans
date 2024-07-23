import random
from typing import Tuple, Dict, Any

import math
import pyscipopt as scip
from balans.utils import Constants


def lp_solve(model) -> Tuple[Dict[Any, float], float]:
    # Solve LP relaxation and save it
    int_index = []
    bin_index = []
    count = 0
    variables = model.getVars()
    for var in variables:
        if var.vtype() == Constants.integer:
            model.chgVarType(var, Constants.continuous)
            int_index.append(count)
        if var.vtype() == Constants.binary:
            model.chgVarType(var, Constants.continuous)
            bin_index.append(count)
        count += 1

    model.optimize()
    lp_index_to_val, lp_obj_val = get_index_to_val_and_objective(model)

    # Get back the original model
    model.freeTransform()
    count = 0
    for var in variables:
        if count in int_index:
            model.chgVarType(var, Constants.integer)
        if count in bin_index:
            model.chgVarType(var, Constants.binary)
        count += 1

    return lp_index_to_val, lp_obj_val


def get_random_solution(model):
    org_objective = model.getObjective()
    variables = model.getVars()
    objective = scip.Expr()
    for var in variables:
        coeff = random.uniform(-1, 1)
        if coeff != 0:
            objective += coeff * var
    objective.normalize()
    model.setObjective(objective, Constants.minimize)
    model.setParam("limits/bestsol", 1)
    model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

    # Solve
    model.optimize()

    r1_index_to_val, r1_obj_val = get_index_to_val_and_objective(model)

    # Get back the original model
    model.freeTransform()
    model.setParam("limits/bestsol", -1)
    model.setObjective(org_objective, Constants.minimize)
    model.setHeuristics(scip.SCIP_PARAMSETTING.DEFAULT)

    return r1_index_to_val, r1_obj_val


def is_discrete(var_type) -> bool:
    return var_type in (Constants.binary, Constants.integer)


def is_binary(var_type) -> bool:
    return var_type in Constants.binary


def get_index_to_val_and_objective(model) -> Tuple[Dict[Any, float], float]:
    # we check if the optimized model has solutions, feasible, and is in the solved state
    if model.getNSols() == 0 or model.getStatus() == "infeasible" or (model.getStage() != 9 and model.getStage() != 10):
        return dict(), 9999999
    else:
        index_to_val = dict([(var.getIndex(), model.getVal(var)) for var in model.getVars()])
        obj_value = model.getObjVal()
        return index_to_val, obj_value


def split_binary_vars(variables, binary_indexes, index_to_val):
    zero_binary_vars = []
    one_binary_vars = []
    for var in variables:
        if var.getIndex() in binary_indexes:
            if math.isclose(index_to_val[var.getIndex()], 0.0):
                zero_binary_vars.append(var)
            else:
                one_binary_vars.append(var)

    return zero_binary_vars, one_binary_vars
