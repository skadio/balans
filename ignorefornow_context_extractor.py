# Class for extract static features

import pandas as pd
from typing import List, Union
import numpy as np
import pyscipopt as scip



class BaseExtractor:
    def __init__(self, problem_instance_file: str) -> None:
        self.model = scip.Model()
        self.model.hideOutput()
        self.model.readProblem(problem_instance_file)


class FeatureExtractor(BaseExtractor):
    def __init__(self, problem_instance_file: str) -> None:
        super().__init__(problem_instance_file)

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

    def extract_variable_features(self):
        varbls = self.model.getVars()
        var_types = [v.vtype() for v in varbls]
        lbs = [v.getLbGlobal() for v in varbls]
        ubs = [v.getUbGlobal() for v in varbls]

        type_mapping = {"BINARY": 0, "INTEGER": 1, "IMPLINT": 2, "CONTINUOUS": 3}
        var_types_numeric = [type_mapping.get(t, 0) for t in var_types]

        variable_features = pd.DataFrame({
            'var_type': var_types_numeric,  # Use the converted numeric representation
            'var_lb': lbs,
            'var_ub': ubs
        })

        variable_features = variable_features.astype({'var_type': int, 'var_lb': float, 'var_ub': float})

        return variable_features

    def extract_constraint_features(self):
        conss = self.model.getConss()
        constraint_data = []
        directions = []
        max_length = 0

        for i, c in enumerate(conss):
            lhs = self.model.getLhs(c)
            rhs = self.model.getRhs(c)
            coefs = self.model.getValsLinear(c)

            if isinstance(lhs, float):
                lhs = [lhs]
            if isinstance(rhs, float):
                rhs = [rhs]
            if isinstance(coefs, float):
                coefs = [coefs]

            max_length = max(max_length, len(lhs), len(rhs), len(coefs))
            constraint_data.append((lhs, rhs, coefs))
            cons_direction = {
                '==': rhs == lhs,
                '<=': rhs != float('inf') and lhs == float('-inf'),
                '>=': lhs != float('inf') and lhs != float('-inf')
            }

            directions.append([d for d, v in cons_direction.items() if v][0])

        constraint_features = []
        for lhs, rhs, coefs in constraint_data:
            feature = {
                'lhs': lhs[0],
                'rhs': rhs[0],
                'cons_coefs': coefs
            }
            constraint_features.append(feature)

        constraint_features = pd.DataFrame(constraint_features)
        constraint_signs = pd.DataFrame({'cons_direction': directions})

        return constraint_features, constraint_signs

    def get_sense(self) -> str:
        sense = self.model.getObjectiveSense()
        return sense

    def extract_bipartite_graph(self):
        variables = self.model.getVars()
        constraints = self.model.getConss()

        variable_names = {}
        constraint_names = {}
        edge_data = []

        # Assign names to variables
        for i, var in enumerate(variables):
            var_name = f'var_{i}'
            variable_names[i] = var_name

        # Assign names to constraints
        for i, cons in enumerate(constraints):
            cons_name = f'cons_{i}'
            constraint_names[i] = cons_name

            cons_linear = self.model.getValsLinear(cons)
            for idx, val in enumerate(cons_linear):
                if val != 0:
                    edge_data.append({
                        'variable': variable_names[idx],
                        'constraint': cons_name,
                        'coefficient': val
                    })

        bipartite_graph = pd.DataFrame(edge_data)

        return bipartite_graph
