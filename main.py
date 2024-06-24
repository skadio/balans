import os
import pickle

from alns.select import MABSelector
from alns.accept import HillClimbing
from alns.stop import MaxIterations, MaxRuntime
from pyscipopt import Model
import pyscipopt
import time

# Contextual multi-armed bandits
from mabwiser.mab import LearningPolicy

# Meta-solver built on top of SCIP
from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants

instance_path = "data/mk/mk_58.lp"
# seed = 89
# # Balans
# balans = Balans(destroy_ops=[DestroyOperators.Crossover,
#                              DestroyOperators.Dins,
#                              DestroyOperators.Mutation,
#                              DestroyOperators.Local_Branching,
#                              DestroyOperators.Rins,
#                              DestroyOperators.Proximity,
#                              DestroyOperators.Rens,
#                              DestroyOperators.Zero_Objective],
#                 repair_ops=[RepairOperators.Repair],
#                 selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=8, num_repair=1,
#                                      learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50), seed=seed),
#                 accept=HillClimbing(),
#                 stop=MaxRuntime(20),
#                 seed=seed)
#
# # Run
# result = balans.solve(instance_path)
#
# # print("Best solution:", result.best_state.solution())
# print("Best solution objective:", result.best_state.objective())
#
# r = [result.statistics.objectives, result.statistics.runtimes, dict(result.statistics.destroy_operator_counts)]
# with open("result.pl", "wb") as fp:
#     pickle.dump(r, fp)
#
# with open("result.pl", "rb") as fp:
#     b = pickle.load(fp)
#
# print(b)

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
    return event.scip_log


#Check for optimality
model = Model("scip")
model.readProblem(instance_path)
r = run_vanilla_scip(model, 36)
r[0] = r[0][1:]
r[1] = r[1][1:]
with open("result.pl", "wb") as fp:
    pickle.dump(r, fp)

with open("result.pl", "rb") as fp:
    b = pickle.load(fp)

print(b)
