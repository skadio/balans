import os
import pickle
import argparse

from alns.select import MABSelector, RandomSelect
from alns.accept import HillClimbing, AlwaysAccept
from alns.stop import MaxIterations, MaxRuntime
from pyscipopt import Model
import pyscipopt
import time

# Contextual multi-armed bandits
from mabwiser.mab import LearningPolicy

# Meta-solver built on top of SCIP
from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants

seed = Constants.default_seed
limit = 600
approach_to_solver_dict = {
    "crossover": Balans(destroy_ops=[DestroyOperators.Crossover],
                repair_ops=[RepairOperators.Repair],
                selector=RandomSelect(num_destroy=1, num_repair=1),
                accept=HillClimbing(),
                stop=MaxRuntime(limit),
                seed=seed),
    "local_branching": Balans(destroy_ops=[DestroyOperators.Local_Branching],
                repair_ops=[RepairOperators.Repair],
                selector=RandomSelect(num_destroy=1, num_repair=1),
                accept=HillClimbing(),
                stop=MaxRuntime(limit),
                seed=seed),
    "mutation": Balans(destroy_ops=[DestroyOperators.Mutation],
                repair_ops=[RepairOperators.Repair],
                selector=RandomSelect(num_destroy=1, num_repair=1),
                accept=HillClimbing(),
                stop=MaxRuntime(limit),
                seed=seed),
    "proximity": Balans(destroy_ops=[DestroyOperators.Proximity],
                repair_ops=[RepairOperators.Repair],
                selector=RandomSelect(num_destroy=1, num_repair=1),
                accept=HillClimbing(),
                stop=MaxRuntime(limit),
                seed=seed),
    "random": Balans(destroy_ops=[DestroyOperators.Zero_Objective],
                repair_ops=[RepairOperators.Repair],
                selector=RandomSelect(num_destroy=1, num_repair=1),
                accept=HillClimbing(),
                stop=MaxRuntime(limit),
                seed=seed),
    "rins": Balans(destroy_ops=[DestroyOperators.Rins],
                repair_ops=[RepairOperators.Repair],
                selector=RandomSelect(num_destroy=1, num_repair=1),
                accept=HillClimbing(),
                stop=MaxRuntime(limit),
                seed=seed),
    "all_EpsilonGreedy": Balans(destroy_ops=[DestroyOperators.Crossover,
                             DestroyOperators.Dins,
                             DestroyOperators.Mutation,
                             DestroyOperators.Local_Branching,
                             DestroyOperators.Rins,
                             DestroyOperators.Proximity,
                             DestroyOperators.Rens,
                             DestroyOperators.Zero_Objective],
                repair_ops=[RepairOperators.Repair],
                selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=8, num_repair=1,
                                     learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50), seed=seed),
                accept=HillClimbing(),
                stop=MaxRuntime(limit),
                seed=seed),
    "all_Softmax": Balans(destroy_ops=[DestroyOperators.Crossover,
                             DestroyOperators.Dins,
                             DestroyOperators.Mutation,
                             DestroyOperators.Local_Branching,
                             DestroyOperators.Rins,
                             DestroyOperators.Proximity,
                             DestroyOperators.Rens,
                             DestroyOperators.Zero_Objective],
                repair_ops=[RepairOperators.Repair],
                selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=8, num_repair=1,
                                     learning_policy=LearningPolicy.Softmax(), seed=seed),
                accept=HillClimbing(),
                stop=MaxRuntime(limit),
                seed=seed),
    "all_UCB": Balans(destroy_ops=[DestroyOperators.Crossover,
                             DestroyOperators.Dins,
                             DestroyOperators.Mutation,
                             DestroyOperators.Local_Branching,
                             DestroyOperators.Rins,
                             DestroyOperators.Proximity,
                             DestroyOperators.Rens,
                             DestroyOperators.Zero_Objective],
                repair_ops=[RepairOperators.Repair],
                selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=8, num_repair=1,
                                     learning_policy=LearningPolicy.UCB1(), seed=seed),
                accept=HillClimbing(),
                stop=MaxRuntime(limit),
                seed=seed)
}

class MyEvent(pyscipopt.Eventhdlr):
    def eventinit(self):
        print("init event")
        self._start_time = time.monotonic()
        self.scip_log = [[],[]]
        self.start_time = time.monotonic()
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        print("exit event")
        #self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        print("exec event")
        self.end_time = time.monotonic()
        sol = self.model.getBestSol()
        obj = self.model.getSolObjVal(sol)
        self.scip_log[0].append(obj)
        self.scip_log[1].append(self.end_time - self.start_time)
        self.start_time = self.end_time


def run_vanilla_scip(model, time):
    model = model.__repr__.__self__
    event = MyEvent()
    model.includeEventhdlr(
        event,
        "",
        ""
    )
    model.setParam("limits/time", time)
    model.optimize()
    event.scip_log[0] = event.scip_log[0][1:]
    event.scip_log[1] = event.scip_log[1][1:]
    return event.scip_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain")
    parser.add_argument("--instance")
    args = parser.parse_args()

    approaches = ["crossover", "local_branching", "mutation", "proximity", "random", "rins",
                  "all_EpsilonGreedy", "all_Softmax", "all_UCB"]

    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + args.domain):
        os.mkdir('results/' + args.domain)
    if not os.path.exists('results/' + args.domain + '/scip/'):
        os.mkdir('results/' + args.domain + '/scip/')

 #   Collect runtime on vanilla scip
    model = Model("scip")
    model.readProblem(args.instance)
    r = run_vanilla_scip(model, limit)
    with open('results/' + args.domain + '/scip/' + args.instance.split("/")[-1], "wb") as fp:
        pickle.dump(r, fp)

    for approach in approaches:
        if not os.path.exists('results/' + args.domain + '/' + approach):
            os.mkdir('results/' + args.domain + '/' + approach)
        solver = approach_to_solver_dict[approach]
        result = solver.solve(args.instance)
        r = [result.statistics.objectives, result.statistics.runtimes, dict(result.statistics.destroy_operator_counts)]
        with open('results/' + args.domain + '/' + approach + '/' + args.instance.split("/")[-1], "wb") as fp:
            pickle.dump(r, fp)

