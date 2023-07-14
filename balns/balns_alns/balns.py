#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pyscipopt as scip
import pandas as pd
from typing import List, Union
from utils import ProblemState

class BaseOperator:
    def __init__(self, problem_instance_file: str) -> None:
        self.model = scip.Model()
        self.model.hideOutput()
        self.model.readProblem(problem_instance_file)


class OperatorExtractor(BaseOperator):
    def __init__(self, problem_instance_file: str) -> None:
        super().__init__(problem_instance_file)
        self.var_features = self.extract_variable_features()

    def init_sol(self) -> ProblemState:
        # solution gap is less than %50  > STOP, terrible but, good start.
        self.model.setParam("limits/gap", 0.50)
        self.model.setParam('limits/time', 30)
        self.model.optimize()
        solution = []
        for v in self.model.getVars():
            if v.name != "n":
                solution.append(self.model.getVal(v))
        solution = np.array(solution)
        len_sol = len(solution)
        print("init sol", self.model.getObjVal())

        state = ProblemState(solution, self.model)

        return state

    def lp_relax(self):

        """
        Gets and Solves LP relaxed version of the same problem

        Returns
        -------
        objective value=float
        solution =array
        len_sol=int
        """

        for v in self.model.getVars():
            # Continuous relaxation of the problem
            self.model.chgVarType(v, 'CONTINUOUS')
        self.model.optimize()
        solution = []
        for v in self.model.getVars():
            if v.name != "n":
                solution.append(self.model.getVal(v))

        solution = np.array(solution)
        len_sol = len(solution)
        return self.model.getObjVal(), solution, len_sol

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

    def extract_feature(self) -> Union[List[Union[int, float]], np.ndarray, pd.Series, pd.DataFrame]:
        """
        Extracts features from MIP instances

        Returns
        -------
        feature_features : Union[List[Union[int, float]], np.ndarray, pd.Series, pd.DataFrame]
            The extracted features
        """
        # Extract variable and constraints features
        variable_features = self.extract_variable_features()
        constraint_features, constraint_signs = self.extract_constraint_features()
        objective_sense = self.get_sense()

        # Create a DataFrame with the desired format for MABSelector
        feature_df = pd.concat([variable_features, constraint_features, constraint_signs], axis=1)
        feature_df.insert(0, 'objective_sense', objective_sense)
        feature_df.loc[1:, 'objective_sense'] = np.nan
        feature_df.fillna('nan', inplace=True)
        feature_df.reset_index(drop=True, inplace=True)

        return feature_df

    def get_var_features(self):
        return self.var_features

    def create_sol(self):
        return self.model.createSol()

    def forget_sol(self, create_solution):
        self.model.freeSol(create_solution)

