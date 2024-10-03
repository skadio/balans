import abc
from typing import Dict, List, Tuple, Any

from balans.utils import Constants


class _BaseMIP(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, seed: int):
        self.seed = seed

        # Model, variables, objective
        self.model = None
        self.variables = None
        self.org_objective_fn = None
        self.org_objective_sense = None
        self.is_obj_sense_changed = False       # we always minimize

        # These are used for incremental solving
        self.constraints = []
        self.proximity_z = None         # Used in proximity
        self.is_obj_transformed = False # Used in proximity and random objective

    @abc.abstractmethod
    def calc_obj_value(self, index_to_val) -> float:
        """ Given a solution, return objective value """
        pass

    @abc.abstractmethod
    def extract_indexes(self) -> Tuple[List[Any], List[Any], List[Any]]:
        """ Return discrete, binary, and integer index lists"""
        pass

    @abc.abstractmethod
    def extract_lp(self, discrete_indexes) -> Tuple[Dict[Any, float], float, List[Any]]:
        """ Return lp index_to_val, lp objective value, and the floating discrete indexes """
        pass

    @abc.abstractmethod
    def fix_vars(self, index_to_val, skip_indexes=None) -> None:
        """ Add constraints to fix variables to given values except skipped vars"""
        pass

    @abc.abstractmethod
    def dins(self, index_to_val, dins_set, lp_index_to_val) -> None:
        pass

    @abc.abstractmethod
    def local_branching(self, index_to_val, local_branching_size, binary_indexes) -> None:
        pass

    @abc.abstractmethod
    def proximity(self, index_to_val, obj_val, proximity_delta, binary_indexes) -> None:
        pass

    @abc.abstractmethod
    def rens(self, index_to_val, rens_float_set, lp_index_to_val) -> None:
        pass

    @abc.abstractmethod
    def random_objective(self) -> None:
        pass

    @abc.abstractmethod
    def solve_and_undo(self, time_limit_in_sc=None, solution_limit=None) -> Tuple[Dict[Any, float], float]:
        """
        Solve with the given time and solution limit, return the solution index_to_val and obj value
        Make sure to undo the solve and clear the constraints, promixity z, and reset objective.
        """
        pass

    @abc.abstractmethod
    def solve_random_and_undo(self, time_limit_in_sc=None) -> Tuple[Dict[Any, float], float]:
        """
        Solve with the given time limit and return a random solution index_to_val and obj value
        Make sure to undo the solve operation.
        """
        pass


def create_mip_solver(instance_path: str, seed: int = Constants.default_seed,
                      mip_solver_str: str = Constants.default_solver) -> _BaseMIP:
    """ Returns a mip model of the given solver type for the given instance

        Parameters
        ----------
        instance_path: str
            the path to mip instance
        seed: int
            the seed to pass to mip model
        mip_solver_str : string
            the type of the mip model, scip or gurobi

        Returns
        -------
        out : _BaseMIP
            A MIP Solver object that implements the base solver class
    """
    from balans.solver_scip import _SCIP
    from balans.solver_gurobi import _Gurobi
    mip_factory = {Constants.scip_solver: _SCIP, Constants.gurobi_solver: _Gurobi}

    return mip_factory.get(mip_solver_str)(instance_path, seed)
