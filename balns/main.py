import numpy as np
from balns import OperatorExtractor
from mutation import _Mutation
from crossover import _Crossover


def run_mip_operator_extractor(instance_path):

    operator_extractor = OperatorExtractor(problem_instance_file=instance_path)
    operator_extractor_init = OperatorExtractor(problem_instance_file=instance_path)
    operator_extractor_init2 = OperatorExtractor(problem_instance_file=instance_path)

    # init solutions
    init_value, solution_init, n = operator_extractor_init.init_sol()
    init_value2, solution_init2, n = operator_extractor_init2.init_sol()

    # lp relaxed value
    lp_relaxed_value, solution, n = operator_extractor.lp_relax()
    #Get MIP instance model, variable features (e.g., binary, discrete or cont) and objective (min or max)
    model = operator_extractor.get_model()
    var_features = operator_extractor.get_var_features()
    sense1 = operator_extractor.get_sense()

    ## Call Crossover and Mutation
    cross_sol2 = _Crossover(solution_init, solution_init2, model, var_features, lp_relaxed_value, init_value,
                            init_value2, sense1)
    cross_sol_next = cross_sol2.crossover_op()
    print("Crossover Solution:", cross_sol_next.get_solution())

    mut_sol2 = _Mutation(solution_init, model, var_features, lp_relaxed_value, init_value, sense1)
    mut_sol_next = mut_sol2.mutation_op()
    print("Mutation Solution:", mut_sol_next.get_solution())

    return mut_sol_next.get_solution(), cross_sol_next.get_solution()


if __name__ == "__main__":
    instance_path = "tests/data/neos-5140963-mincio.mps.gz"

    # Return next solutions
    solution_mut, solution_cross = run_mip_operator_extractor(instance_path)
