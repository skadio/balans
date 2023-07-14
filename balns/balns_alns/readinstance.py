# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pyscipopt as scip


class BaseRead:
    def __init__(self, problem_instance_file: str) -> None:
        self.model = scip.Model()
        self.model.hideOutput()
        self.model.readProblem(problem_instance_file)


class ReadInstance(BaseRead):
    def __init__(self, problem_instance_file: str) -> None:
        super().__init__(problem_instance_file)
        self.var_features = self.extract_variable_features()



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

    def extract_variable_features(self):
        varbls = self.model.getVars()
        var_types = [v.vtype() for v in varbls]
        lbs = [v.getLbGlobal() for v in varbls]
        ubs = [v.getUbGlobal() for v in varbls]

        type_mapping = {"BINARY": 0, "INTEGER": 1, "IMPLINT": 2, "CONTINUOUS": 3}
        var_types_numeric = [type_mapping.get(t, 0) for t in var_types]

        variable_features = pd.DataFrame({
            'var_type': var_types_numeric,
            'var_lb': lbs,
            'var_ub': ubs
        })

        variable_features = variable_features.astype({'var_type': int, 'var_lb': float, 'var_ub': float})

        return variable_features  # pd.dataframe