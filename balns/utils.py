# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyscipopt as scip
from balns import OperatorExtractor

class MIPState:
    """
    Generic problem class for the mip problem. It stores the current
    solution as a vector of variables, one for each item.

    Current objective also stored.
    """

    def __init__(self, solution: np.array, model):
        self.solution = solution
        self.model = model

    def objective(self):
        return self.model.getObjVal(self.solution)

    def proxy_objective(self) -> int:
        p = np.random.randint(1, 100, size=len(self.solution))
        return p @ self.model

    def length(self) -> int:
        return len(self.model)

    def get_solution(self):
        return self.solution
