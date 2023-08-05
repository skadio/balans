from typing import Tuple, Dict, Any

import pyscipopt as scip

from balans.utils import Constants


def get_model_and_vars(path, is_verbose=False, has_pre_solve=True,
                       solution_count=None, gap=None, time=None,
                       is_lp_relaxation=False):
    # TODO need to think about what SCIP defaults to use, turn-off SCIP-ALNS?

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


def lp_solve(path) -> Tuple[Dict[Any, float], float]:

    # Build model and variables
    model, variables = get_model_and_vars(path, is_lp_relaxation=True)

    # Solve
    model.optimize()
    index_to_val = get_index_to_val(model)
    obj_value = model.getObjVal()

    # Return solution and objective
    return index_to_val, obj_value


def random_solve(path, gap=Constants.random_gap, time=Constants.random_time) -> Tuple[Dict[Any, float], float]:

    # Build model and variables
    model, variables = get_model_and_vars(path, gap=gap, time=time)

    # Solve
    model.optimize()
    random_index_to_val = get_index_to_val(model)
    random_obj_value = model.getObjVal()

    # Reset problem
    # TODO Why is this needed?
    model.freeProb()

    # Return solution
    return random_index_to_val, random_obj_value


def is_discrete(var_type) -> bool:
    return var_type in (Constants.binary, Constants.integer)


def is_binary(var_type) -> bool:
    return var_type in Constants.binary


def get_index_to_val(model) -> Dict[Any, float]:
    return dict([(var.getIndex(), model.getVal(var)) for var in model.getVars()])


def split_binary_vars(variables, binary_indexes, index_to_val):
    zero_binary_vars = []
    one_binary_vars = []
    for var in variables:
        if var.getIndex() in binary_indexes:
            if index_to_val[var.getIndex()] == 0:
                zero_binary_vars.append(var)
            else:
                one_binary_vars.append(var)

    return zero_binary_vars, one_binary_vars
