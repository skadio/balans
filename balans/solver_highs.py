import math
import random
from typing import Tuple, Dict, Any, List, Sequence

import highspy
import numpy as np
from highspy import Highs, kHighsInf, HighsStatus, ObjSense, HighsVarType as VarType

from balans.base_mip import _BaseMIP
from balans.utils import Constants

# Highs examples:
# https://github.com/ERGO-Code/HiGHS/blob/latest/tests/test_highspy.py
class HighsSolver(_BaseMIP):

    def __init__(self, instance_path: str, seed: int):
        super().__init__(seed)

        self.model = Highs()
        status = self.model.readModel(instance_path)
        if status != HighsStatus.kOk:
            raise RuntimeError(f"HiGHS failed to read '{instance_path}' (status={status}).")

        self.model.setOptionValue("random_seed", seed)

        self.variables = self.model.getCols()  # list[highs_var]
        self.lp = self.model.getLp()

        self.org_coeffs: List[float] = list(self.lp.col_cost_)
        self.org_objective_fn = sum(c * v for c, v in zip(self.org_coeffs, self.variables))
        self.org_objective_sense = self.model.getObjectiveSense()[1]
        self.is_obj_sense_changed = False

        if self.org_objective_sense == ObjSense.kMaximize:
            self.org_coeffs = [-c for c in self.org_coeffs]
            self.org_objective_fn = -self.org_objective_fn
            self.model.minimize(self.org_objective_fn)
            self.is_obj_sense_changed = True

        self.constraints: List[Any] = []
        self.proximity_z = None
        self.is_obj_transformed = False

    def _objective_expression(self):
        return sum(self.lp.col_cost_[i] * v for i, v in enumerate(self.variables))

    @staticmethod
    def _is_discrete(vtype: VarType) -> bool:
        return vtype in (VarType.kBinary, VarType.kInteger)

    @staticmethod
    def _is_binary(vtype: VarType) -> bool:
        return vtype == VarType.kBinary

    def get_obj_value(self, index_to_val: Dict[int, float]) -> float:
        return sum(self.org_coeffs[i] * index_to_val.get(i, 0.0)
                   for i in range(len(self.org_coeffs)))

    def extract_indexes(self) -> Tuple[List[int], List[int], List[int]]:
        discrete, binary, integer = [], [], []
        for v in self.variables:
            if self._is_discrete(v.type):
                discrete.append(v.index)
                (binary if self._is_binary(v.type) else integer).append(v.index)
        return discrete, binary, integer

    def extract_lp(self, discrete_idxs: Sequence[int]):
        lp_vals, lp_obj = self.solve_lp_and_undo()
        floating = [i for i in discrete_idxs if not math.isclose(lp_vals[i] % 1, 0.0)]
        return lp_vals, lp_obj, floating

    def fix_vars(self, idx_to_val: Dict[int, float] | None, skip: set[int] | None = None):
        if not idx_to_val:
            return
        for v in self.variables:
            if v.index not in idx_to_val or (skip and v.index in skip):
                continue
            cid = self.model.addConstr(v == idx_to_val[v.index])
            self.constraints.append(cid)

    def dins(self, idx_to_val, dins_set, lp_vals):
        for v in self.variables:
            i = v.index
            if i in dins_set:
                diff = abs(idx_to_val[i] - lp_vals[i])
                self.constraints.append(self.model.addConstr(v >= lp_vals[i] - diff))
                self.constraints.append(self.model.addConstr(v <= lp_vals[i] + diff))
            else:
                self.constraints.append(self.model.addConstr(v == idx_to_val[i]))

    def local_branching(self, idx_to_val, radius: int, binary_idx):
        zero, one = self.split_binary_vars(self.variables, binary_idx, idx_to_val)
        expr = sum(zero) + sum(1 - v for v in one)
        self.constraints.append(self.model.addConstr(expr <= radius))

    def proximity(self, idx_to_val, obj_val, delta: float, binary_idx):
        self.is_obj_transformed = True
        zero, one = self.split_binary_vars(self.variables, binary_idx, idx_to_val)
        expr_obj = sum(zero) + sum(1 - v for v in one)

        self.proximity_z = self.model.addVariable(lb=0, name="proximity_z")
        rhs = obj_val * (1 - delta) + self.proximity_z
        self.constraints.append(self.model.addConstr(self._objective_expression() <= rhs))

        self.model.minimize(expr_obj + Constants.M * self.proximity_z)

    def rens(self, idx_to_val, rens_set, lp_vals):
        for v in self.variables:
            i = v.index
            if i in rens_set:
                self.constraints.append(self.model.addConstr(v >= math.floor(lp_vals[i])))
                self.constraints.append(self.model.addConstr(v <= math.ceil(lp_vals[i])))
            else:
                self.constraints.append(self.model.addConstr(v == idx_to_val[i]))

    def random_objective(self):
        self.is_obj_transformed = True
        n = len(self.variables)
        keep = max(1, int(0.25 * n))
        chosen = set(random.sample(range(n), keep))

        expr = sum(self.org_coeffs[i] * self.variables[i] for i in chosen)
        self.model.minimize(expr)
        self.model.setOptionValue("solution_limit", 1)
        self.model.setOptionValue("heuristics", 0)

    def solve_and_undo(self, time_limit_sc: float | None = None, sol_limit: int | None = None):
        if time_limit_sc is not None:
            self.model.setOptionValue("time_limit", time_limit_sc)
        if sol_limit is not None:
            self.model.setOptionValue("solution_limit", sol_limit)

        self.model.run()
        vals, obj = self.get_index_to_val_and_objective()
        self._reset_after_solve(time_limit_sc, sol_limit)
        return vals, obj

    def solve_random_and_undo(self, time_limit_sc: float | None = None):
        if time_limit_sc is not None:
            self.model.setOptionValue("time_limit", time_limit_sc)

        self.random_objective()
        self.model.run()
        vals, obj = self.get_index_to_val_and_objective()

        # restore original objective & options
        self.is_obj_transformed = False
        self.model.minimize(self.org_objective_fn)
        self.model.setOptionValue("solution_limit", kHighsInf)
        if time_limit_sc is not None:
            self.model.setOptionValue("time_limit", kHighsInf)

        return vals, obj

    def solve_lp_and_undo(self):
        int_idx, bin_idx = [], []
        for v in self.variables:
            if self._is_binary(v.type):
                bin_idx.append(v.index)
            elif self._is_discrete(v.type):
                int_idx.append(v.index)

        if int_idx:
            self.model.changeColsIntegrality(len(int_idx),
                                             np.array(int_idx, dtype=np.int32),
                                             np.full(len(int_idx), VarType.kContinuous, dtype=np.int32))

        if bin_idx:
            self.model.changeColsIntegrality(len(bin_idx),
                                             np.array(bin_idx, dtype=np.int32),
                                             np.full(len(bin_idx), VarType.kContinuous, dtype=np.int32))

        self.model.run()

        lp_vals, lp_obj = self.get_index_to_val_and_objective()

        if int_idx:
            self.model.changeColsIntegrality(len(int_idx),
                                             np.array(int_idx, dtype=np.int32),
                                             np.full(len(int_idx), VarType.kInteger, dtype=np.int32))
        if bin_idx:
            self.model.changeColsIntegrality(len(bin_idx),
                                             np.array(bin_idx, dtype=np.int32),
                                             np.full(len(bin_idx), VarType.kBinary, dtype=np.int32))

        return lp_vals, lp_obj

    def get_index_to_val_and_objective(self):
        status = self.model.getModelStatus()
        if status in (highspy.HighsModelStatus.kInfeasible, highspy.HighsModelStatus.kUnbounded):
            return {}, float("inf")

        sol = self.model.getSolution()
        idx_to_val = {i: v for i, v in enumerate(sol.col_value)}
        obj_val = self.model.getInfo().objective_function_value
        return idx_to_val, obj_val

    def _reset_after_solve(self, time_limit_sc, sol_limit):
        if time_limit_sc is not None:
            self.model.setOptionValue("time_limit", time_limit_sc)
        if sol_limit is not None:
            self.model.setOptionValue("solution_limit", sol_limit)
