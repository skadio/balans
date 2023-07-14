import numpy as np
import pyscipopt as scip


class BaseRead:
    def __init__(self, problem_instance_file: str) -> None:
        self.model = scip.Model()
        self.model.hideOutput()
        self.model.readProblem(problem_instance_file)
        self.model.setPresolve(scip.SCIP_PARAMSETTING.OFF)


class ReadInstance(BaseRead):
    def __init__(self, problem_instance_file: str) -> None:
        super().__init__(problem_instance_file)
        #self.var_features = self.extract_variable_features()


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
