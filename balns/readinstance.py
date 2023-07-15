import numpy as np
import pyscipopt as scip
from problemstate import ProblemState


class BaseRead:
    def __init__(self, problem_instance_file: str) -> None:
        self.model = scip.Model()
        self.model.hideOutput()
        self.model.readProblem(problem_instance_file)
        self.model.setPresolve(scip.SCIP_PARAMSETTING.OFF)


class ReadInstance(BaseRead):
    def __init__(self, problem_instance_file: str) -> None:
        super().__init__(problem_instance_file)
        # self.var_features = self.extract_variable_features()

    def get_sense(self) -> str:
        """
        Returns
        -------
        objective -> Minimize or Maximize
        """

        sense = self.model.getObjectiveSense()
        return sense

    def get_model(self):
        return self.model

    def initial_state(self, gap, time) -> ProblemState:
        # TODO implement a function that returns an initial solution

        # TODO Solve with scip stop at feasible

        # solution gap is less than something  > STOP.
        self.model.setParam("limits/gap", gap)
        self.model.setParam('limits/time', time)
        self.model.optimize()
        solution_ar = []
        for v in self.model.getVars():
            if v.name != "n":
                solution_ar.append(self.model.getVal(v))

        solution = self.model.getBestSol()

        state = ProblemState(solution, self.model)

        return state
