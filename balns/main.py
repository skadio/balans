import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype
import numpy as np
from balns import OperatorExtractor

#from utils import MIPState

from mutation import _Mutation

def run_mip_operator_extractor(instance_path):
    SEED = 42
    rnd_state = np.random.RandomState(SEED)
    # print(rnd_state)
    # In our operators this is delta
    # Percentage of items to remove in each iteration
    delta = .25  # change this to delta
    n = 196
    operator_extractor = OperatorExtractor(problem_instance_file=instance_path)

    lp_relaxed_value, solution, n = operator_extractor.LP_relax()
    model = operator_extractor.Model()
    var_features = operator_extractor.get_var_features()

    #print("LP RELAXED Solution:", solution)
    #print("LP RELAXED Objective Value:", lp_relaxed_value)
    print("Num Var:", n)

    # # Read the LP solution

    #lp_sol = MIPState(solution,model)
    #lp_sol.objective()

    #mut_sol = MIPState(solution,model) #Initial MIP State

    #init_sol = np.zeros(n)
    #init_sol[0]=9


    mut_sol2 = _Mutation(solution,model,var_features)

    mut_sol_next =mut_sol2.mutation_op()
    print("Mutation Solution:",mut_sol_next.solution)

    return mut_sol_next.solution

# Terrible - but simple - first solution, where only the first item is
# selected.
# init_sol = MIPState(np.zeros(n))
# init_sol.x[0] = 1

# init_sol.objective()


if __name__ == "__main__":
    instance_path = "data/neos-5140963-mincio.mps.gz"

    # Create MIP instance and LP relaxed Solution
    solution = run_mip_operator_extractor(instance_path)

