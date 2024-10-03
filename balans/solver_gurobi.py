from typing import Tuple, Dict, Any, List

from balans.base_mip import _BaseMIP


class _Gurobi(_BaseMIP):

    def __init__(self, instance_path: str, seed: int):
        super().__init__(seed)
        # TODO implement gurobi model
        self.model = None

    def calc_obj_value(self, index_to_val) -> float:
        pass

    def extract_indexes(self) -> Tuple[List[Any], List[Any], List[Any]]:
        pass

    def extract_lp(self, discrete_indexes) -> Tuple[Dict[Any, float], float, List[Any]]:
        pass

    def fix_vars(self, index_to_val, skip_indexes=None) -> None:
        pass

    def dins(self, index_to_val, dins_set, lp_index_to_val) -> None:
        pass

    def local_branching(self, index_to_val, local_branching_size, binary_indexes) -> None:
        pass

    def proximity(self, index_to_val, obj_val, proximity_delta, binary_indexes) -> None:
        pass

    def rens(self, index_to_val, rens_float_set, lp_index_to_val) -> None:
        pass

    def random_objective(self) -> None:
        pass

    def solve_and_undo(self, time_limit_in_sc=None, solution_limit=None) -> Tuple[Dict[Any, float], float]:
        pass

    def solve_random_and_undo(self) -> Tuple[Dict[Any, float], float]:
        pass
