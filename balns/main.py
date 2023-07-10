import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype
import numpy as np
from balns import OperatorExtractor

#from utils import MIPState

from mutation import _Mutation
from crossover import _Crossover

def run_mip_operator_extractor(instance_path):
    SEED = 42
    rnd_state = np.random.RandomState(SEED)

    operator_extractor = OperatorExtractor(problem_instance_file=instance_path)
    operator_extractor_init = OperatorExtractor(problem_instance_file=instance_path)

    operator_extractor_init2 = OperatorExtractor(problem_instance_file=instance_path)

    #init solutions
    # Terrible - but simple - first solution, where only the first item is
    # selected.

    init_value, solution_init,n = operator_extractor_init.Init_Sol()

    init_value2, solution_init2,n = operator_extractor_init2.Init_Sol()

    #lp relaxed value
    lp_relaxed_value, solution, n = operator_extractor.LP_relax()
    #solution_init = operator_extractor.Init_Sol()
    model = operator_extractor.Model()
    var_features = operator_extractor.get_var_features()
    sense1 = operator_extractor.get_sense()


    cross_sol2 = _Crossover(solution_init,solution_init2,model,var_features,lp_relaxed_value,init_value,init_value2,sense1)

    cross_sol_next =cross_sol2.crossover_op()
    print("Crossover Solution:",cross_sol_next.solution)

    #return mut_sol_next.solution

    mut_sol2 = _Mutation(solution_init,model,var_features,lp_relaxed_value,init_value,sense1)

    mut_sol_next =mut_sol2.mutation_op()
    print("Mutation Solution:",mut_sol_next.solution)

    return mut_sol_next.solution




if __name__ == "__main__":
    instance_path = "data/neos-5140963-mincio.mps.gz"

    # Create MIP instance and LP relaxed Solution
    solution = run_mip_operator_extractor(instance_path)

