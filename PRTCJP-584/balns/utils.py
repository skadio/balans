# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
:Author:

This module provides a number of constants and helper functions.
"""

import numpy as np
from balns import OperatorExtractor

class MIPState:
    """
    Solution class for the mip problem. It stores the current
    solution as a vector of variables, one for each item.

    Current objective also stored.
    """

    def __init__(self, solution :np.array, model):
        #self.x = x
        self.solution=solution
        self.model = model
        # print(x)

    def objective(self) -> int:
        # return model.getSolObjVal(self.x)
        return self.model.getObjVal(self.solution)

    def proxy_objective(self) -> int:
        # return model.getSolObjVal(self.x)
        p = np.random.randint(1, 100, size=len(self.solution))
        return p @ self.model

    def lenght(self) -> int:
        return len(self.model)