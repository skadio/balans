import numpy as np
from mabwiser.mab import LearningPolicy
from alns import ALNS
from alns.accept import *
from alns.select import *
from alns.stop import *
from problemstate import ProblemState
from readinstance import ReadInstance
from mutation import mutation_op,to_destroy_mut,find_discrete,extract_variable_features,repair_op
SEED = 42
np.random.seed(SEED)

def initial_state(instance_path, gap, time) -> ProblemState:
    # TODO implement a function that returns an initial solution

    # TODO Solve with scip stop at feasible
    instance = ReadInstance(problem_instance_file=instance_path)
    model = instance.get_model()

    # solution gap is less than %50  > STOP, terrible but, good start.
    model.setParam("limits/gap", gap)
    model.setParam('limits/time', time)
    model.optimize()
    solution = []
    for v in model.getVars():
        if v.name != "n":
            solution.append(model.getVal(v))
    # solution = np.array(solution)
    len_sol = len(solution)
    solution = model.getBestSol()
    # print("init sol", self.model.getObjVal())

    # solution2=model.createSol() #scip in icinde tanimli
    # solution3=model.createSol()
    state = ProblemState(solution, model)

    return state


if __name__ == "__main__":
    instance_path = "neos-5140963-mincio.mps.gz"

    # Terrible - but simple - two first solution, where only the first item is
    # selected.

    #Time =30 and gap limit = 50 percent close the solution
    init_sol = initial_state(instance_path, 0.50, 30)

    # print(init_sol.transform_solution_to_array())

    init_sol2 = initial_state(instance_path, 0.75, 30)
    #print("Initial Feasible Solution:", init_sol2.transform_solution_to_array())
    print("Initial Feasible Solution:", init_sol2.transform_solution_to_array2())

def make_alns() -> ALNS:
    rnd_state = np.random.RandomState(SEED)
    alns = ALNS(rnd_state)
    alns.add_destroy_operator(mutation_op)
    alns.add_repair_operator(repair_op)
    return alns

accept = HillClimbing()



#MABSelector
select = MABSelector(scores=[5, 2, 1, 0.5],
                     num_destroy=1,
                     num_repair=1,
                     learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))

alns = make_alns()
res = alns.iterate(init_sol, select, accept, MaxIterations(2))

print(f"Found solution with objective {res.best_state.objective()}.")