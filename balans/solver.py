import os
from typing import List, Optional, Dict
from typing import NamedTuple

import numpy as np
from alns.ALNS import ALNS
from alns.Result import Result
from alns.accept import LateAcceptanceHillClimbing, NonLinearGreatDeluge, AlwaysAccept
from alns.accept import MovingAverageThreshold, GreatDeluge, HillClimbing
from alns.accept import RecordToRecordTravel, SimulatedAnnealing, RandomAccept
from alns.select import AlphaUCB, MABSelector, RandomSelect, RouletteWheel, SegmentedRouletteWheel
from alns.stop import MaxIterations, MaxRuntime, NoImprovement, StoppingCriterion

from balans.base_instance import _Instance
from balans.base_mip import create_mip_solver
from balans.base_state import _State
from balans.destroy.crossover import crossover
from balans.destroy.dins import dins
from balans.destroy.local_branching import local_branching_05, local_branching_10, local_branching_15, local_branching_20, local_branching_25, local_branching_30, local_branching_35, local_branching_40,  local_branching_45,  local_branching_50, local_branching_55,  local_branching_60,  local_branching_65,  local_branching_70,  local_branching_75,  local_branching_80, local_branching_85,  local_branching_90,  local_branching_95
from balans.destroy.local_branching_relax import local_branching_relax_10, local_branching_relax_25
from balans.destroy.mutation import mutation_05, mutation_10, mutation_15, mutation_20, mutation_25, mutation_30, mutation_35, mutation_40, mutation_45, mutation_50, mutation_55, mutation_60, mutation_65, mutation_70, mutation_75, mutation_80, mutation_85, mutation_90, mutation_95
from balans.destroy.proximity import proximity_05, proximity_10, proximity_15, proximity_20, proximity_25, proximity_30, proximity_35, proximity_40, proximity_45, proximity_50, proximity_55, proximity_60, proximity_65, proximity_70, proximity_75, proximity_80, proximity_85, proximity_90, proximity_95
from balans.destroy.random_objective import random_objective
from balans.destroy.rens import rens_05, rens_10, rens_15, rens_20, rens_25, rens_30, rens_35, rens_40, rens_45, rens_50, rens_55, rens_60, rens_65, rens_70, rens_75, rens_80, rens_85, rens_90, rens_95
from balans.destroy.rins import rins_05, rins_10, rins_15, rins_20, rins_25, rins_30, rins_35, rins_40, rins_45, rins_50, rins_55, rins_60, rins_65, rins_70, rins_75, rins_80, rins_85, rins_90, rins_95
from balans.repair.repair import repair
from balans.utils import Constants, check_false, check_true, create_rng


class DestroyOperators(NamedTuple):
    Crossover = crossover
    Dins = dins
    Local_Branching_05 = local_branching_05
    Local_Branching_10 = local_branching_10
    Local_Branching_15 = local_branching_15
    Local_Branching_20 = local_branching_20
    Local_Branching_25 = local_branching_25
    Local_Branching_30 = local_branching_30
    Local_Branching_35 = local_branching_35
    Local_Branching_40 = local_branching_40
    Local_Branching_45 = local_branching_45
    Local_Branching_50 = local_branching_50
    Local_Branching_55 = local_branching_55
    Local_Branching_60 = local_branching_60
    Local_Branching_65 = local_branching_65
    Local_Branching_70 = local_branching_70
    Local_Branching_75 = local_branching_75
    Local_Branching_80 = local_branching_80
    Local_Branching_85 = local_branching_85
    Local_Branching_90 = local_branching_90
    Local_Branching_95 = local_branching_95

    Local_Branching_Relax_10 = local_branching_relax_10
    Local_Branching_Relax_25 = local_branching_relax_25

    Mutation_05 = mutation_05
    Mutation_10 = mutation_10
    Mutation_15 = mutation_15
    Mutation_20 = mutation_20
    Mutation_25 = mutation_25
    Mutation_30 = mutation_30
    Mutation_35 = mutation_35
    Mutation_40 = mutation_40
    Mutation_45 = mutation_45
    Mutation_50 = mutation_50
    Mutation_55 = mutation_55
    Mutation_60 = mutation_60
    Mutation_65 = mutation_65
    Mutation_70 = mutation_70
    Mutation_75 = mutation_75
    Mutation_80 = mutation_80
    Mutation_85 = mutation_85
    Mutation_90 = mutation_90
    Mutation_95 = mutation_95

    Proximity_05 = proximity_05
    Proximity_10 = proximity_10
    Proximity_15 = proximity_15
    Proximity_20 = proximity_20
    Proximity_25 = proximity_25
    Proximity_30 = proximity_30
    Proximity_35 = proximity_35
    Proximity_40 = proximity_40
    Proximity_45 = proximity_45
    Proximity_50 = proximity_50
    Proximity_55 = proximity_55
    Proximity_60 = proximity_60
    Proximity_65 = proximity_65
    Proximity_70 = proximity_70
    Proximity_75 = proximity_75
    Proximity_80 = proximity_80
    Proximity_85 = proximity_85
    Proximity_90 = proximity_90
    Proximity_95 = proximity_95

    Rens_05 = rens_05
    Rens_10 = rens_10
    Rens_15 = rens_15
    Rens_20 = rens_20
    Rens_25 = rens_25
    Rens_30 = rens_30
    Rens_35 = rens_35
    Rens_40 = rens_40
    Rens_45 = rens_45
    Rens_50 = rens_50
    Rens_55 = rens_55
    Rens_60 = rens_60
    Rens_65 = rens_65
    Rens_70 = rens_70
    Rens_75 = rens_75
    Rens_80 = rens_80
    Rens_85 = rens_85
    Rens_90 = rens_90
    Rens_95 = rens_95

    Rins_05 = rins_05
    Rins_10 = rins_10
    Rins_15 = rins_15
    Rins_20 = rins_20
    Rins_25 = rins_25
    Rins_30 = rins_30
    Rins_35 = rins_35
    Rins_40 = rins_40
    Rins_45 = rins_45
    Rins_50 = rins_50
    Rins_55 = rins_55
    Rins_60 = rins_60
    Rins_65 = rins_65
    Rins_70 = rins_70
    Rins_75 = rins_75
    Rins_80 = rins_80
    Rins_85 = rins_85
    Rins_90 = rins_90
    Rins_95 = rins_95

    Random_Objective = random_objective


class RepairOperators(NamedTuple):
    Repair = repair


# Type Declarations
DestroyType = (type(DestroyOperators.Crossover),
               type(DestroyOperators.Dins),
               type(DestroyOperators.Local_Branching_05),
               type(DestroyOperators.Local_Branching_10),
               type(DestroyOperators.Local_Branching_15),
               type(DestroyOperators.Local_Branching_20),
               type(DestroyOperators.Local_Branching_25),
               type(DestroyOperators.Local_Branching_30),
               type(DestroyOperators.Local_Branching_35),
               type(DestroyOperators.Local_Branching_40),
               type(DestroyOperators.Local_Branching_45),
               type(DestroyOperators.Local_Branching_50),
               type(DestroyOperators.Local_Branching_55),
               type(DestroyOperators.Local_Branching_60),
               type(DestroyOperators.Local_Branching_65),
               type(DestroyOperators.Local_Branching_70),
               type(DestroyOperators.Local_Branching_75),
               type(DestroyOperators.Local_Branching_80),
               type(DestroyOperators.Local_Branching_85),
               type(DestroyOperators.Local_Branching_90),
               type(DestroyOperators.Local_Branching_95),
               type(DestroyOperators.Local_Branching_Relax_10),
               type(DestroyOperators.Local_Branching_Relax_25),
               type(DestroyOperators.Mutation_05),
               type(DestroyOperators.Mutation_10),
               type(DestroyOperators.Mutation_15),
               type(DestroyOperators.Mutation_20),
               type(DestroyOperators.Mutation_25),
               type(DestroyOperators.Mutation_30),
               type(DestroyOperators.Mutation_35),
               type(DestroyOperators.Mutation_40),
               type(DestroyOperators.Mutation_45),
               type(DestroyOperators.Mutation_50),
               type(DestroyOperators.Mutation_55),
               type(DestroyOperators.Mutation_60),
               type(DestroyOperators.Mutation_65),
               type(DestroyOperators.Mutation_70),
               type(DestroyOperators.Mutation_75),
               type(DestroyOperators.Mutation_80),
               type(DestroyOperators.Mutation_85),
               type(DestroyOperators.Mutation_90),
               type(DestroyOperators.Mutation_95),
               type(DestroyOperators.Proximity_05),
               type(DestroyOperators.Proximity_10),
               type(DestroyOperators.Proximity_15),
               type(DestroyOperators.Proximity_20),
               type(DestroyOperators.Proximity_25),
               type(DestroyOperators.Proximity_30),
               type(DestroyOperators.Proximity_35),
               type(DestroyOperators.Proximity_40),
               type(DestroyOperators.Proximity_45),
               type(DestroyOperators.Proximity_50),
               type(DestroyOperators.Proximity_55),
               type(DestroyOperators.Proximity_60),
               type(DestroyOperators.Proximity_65),
               type(DestroyOperators.Proximity_70),
               type(DestroyOperators.Proximity_75),
               type(DestroyOperators.Proximity_80),
               type(DestroyOperators.Proximity_85),
               type(DestroyOperators.Proximity_90),
               type(DestroyOperators.Proximity_95),
               type(DestroyOperators.Rens_05),
               type(DestroyOperators.Rens_10),
               type(DestroyOperators.Rens_15),
               type(DestroyOperators.Rens_20),
               type(DestroyOperators.Rens_25),
               type(DestroyOperators.Rens_30),
               type(DestroyOperators.Rens_35),
               type(DestroyOperators.Rens_40),
               type(DestroyOperators.Rens_45),
               type(DestroyOperators.Rens_50),
               type(DestroyOperators.Rens_55),
               type(DestroyOperators.Rens_60),
               type(DestroyOperators.Rens_65),
               type(DestroyOperators.Rens_70),
               type(DestroyOperators.Rens_75),
               type(DestroyOperators.Rens_80),
               type(DestroyOperators.Rens_85),
               type(DestroyOperators.Rens_90),
               type(DestroyOperators.Rens_95),
               type(DestroyOperators.Rins_05),
               type(DestroyOperators.Rins_10),
               type(DestroyOperators.Rins_15),
               type(DestroyOperators.Rins_20),
               type(DestroyOperators.Rins_25),
               type(DestroyOperators.Rins_30),
               type(DestroyOperators.Rins_35),
               type(DestroyOperators.Rins_40),
               type(DestroyOperators.Rins_45),
               type(DestroyOperators.Rins_50),
               type(DestroyOperators.Rins_55),
               type(DestroyOperators.Rins_60),
               type(DestroyOperators.Rins_65),
               type(DestroyOperators.Rins_70),
               type(DestroyOperators.Rins_75),
               type(DestroyOperators.Rins_80),
               type(DestroyOperators.Rins_85),
               type(DestroyOperators.Rins_90),
               type(DestroyOperators.Rins_95),
               type(DestroyOperators.Random_Objective))

RepairType = (type(RepairOperators.Repair))

AcceptType = (MovingAverageThreshold,
              GreatDeluge,
              HillClimbing,
              LateAcceptanceHillClimbing,
              NonLinearGreatDeluge,
              AlwaysAccept,
              RecordToRecordTravel,
              SimulatedAnnealing,
              RandomAccept)

SelectorType = (AlphaUCB,
                MABSelector,
                RandomSelect,
                RouletteWheel,
                SegmentedRouletteWheel)

StopType = (MaxIterations,
            MaxRuntime,
            NoImprovement,
            StoppingCriterion)


class Balans:
    """
    High-Level Architecture:

    From the input MIP file, an Instance() is created.
    The Instance() provides:
        - Seed
        - MIP model
        - LP solution and objective
        - Indices for binary, discrete variables
        - solve(operator_settings)
        - undo_solve()

    From an Instance(), a State() is created.
    The State() provides:
        - Instance
        - Solution
        - Previous solution
        - Operator settings
        - solve_and_update()
            - calls Instance.solve(operator_settings)
    From initial state, ALNS() is created.
    ALNS iterates by calling a pair of destroy_repair on State()

    Operators takes a State()
        - Destroy operators updates States.operator_settings
        - Repair operator calls State.solve_and_update()
    """

    def __init__(self,
                 destroy_ops: List,
                 repair_ops: List,
                 selector: SelectorType,
                 accept: AcceptType,
                 stop: StopType,
                 seed: int = Constants.default_seed,  # The random seed
                 n_jobs: int = 1,  # Number of parallel jobs
                 mip_solver: str = Constants.default_solver  # MIP solver scip/gurobi
                 ):

        # Validate arguments
        self._validate_balans_args(destroy_ops, repair_ops, selector, accept, stop, seed, n_jobs, mip_solver)

        # Parameters
        self.destroy_ops = destroy_ops
        self.repair_ops = repair_ops
        self.selector = selector
        self.accept = accept
        self.stop = stop
        self.seed = seed
        self.n_jobs = n_jobs
        self.mip_solver_str = mip_solver

        # RNG
        self._rng = create_rng(self.seed)
        self.alns_seed = self._rng.randint(0, self.seed)

        # ALNS
        self.destroy_ops = destroy_ops
        self.repair_ops = repair_ops
        self.alns = None

        # Instance and the first solution
        self._instance: Optional[_Instance] = None
        self._initial_index_to_val: Optional[Dict[int, float]] = None
        self._initial_obj_val: Optional[float] = None

    @property
    def instance(self) -> _Instance:
        return self._instance

    @property
    def initial_index_to_val(self) -> Dict[int, float]:
        return self._initial_index_to_val

    @property
    def initial_obj_val(self) -> float:
        return self._initial_obj_val

    def solve(self, instance_path, index_to_val=None) -> Result:
        """
        instance_path: the path to the MIP instance file
        index_to_val: initial (partial) solution to warm start the variables

        Returns
        -------
        Result
            ALNS result object, containing the best solution and some additional statistics.
                result.best_state.solution()
                result.best_state.objective()
        """
        self._validate_solve_args(instance_path)

        # MIP is an instance of _BaseMIP created from given mip instance
        mip = create_mip_solver(instance_path, self.seed, self.mip_solver_str)

        # Create instance with mip model created from mip instance
        self._instance = _Instance(mip, self.seed)

        self._initial_index_to_val, self._initial_obj_val = self._instance.initial_solve(index_to_val=index_to_val)

        print(">>> START objective:", self._initial_obj_val)
        # print(">>> START values:", self._initial_index_to_val)

        # Initial state and solution
        initial_state = _State(self.instance, self.initial_index_to_val, self.initial_obj_val,
                               previous_index_to_val=self.initial_index_to_val)

        # Create ALNS
        self.alns = ALNS(np.random.RandomState(self.alns_seed))

        # Set ALNS operators according to MIP type, and if successful, start iterating ALNS
        if self._set_alns_operators():

            # Iterate ALNS
            result = self.alns.iterate(initial_state, self.selector, self.accept, self.stop)

            # Restore the original objective results, if max problem is reversed
            if self.instance.mip.is_obj_sense_changed:
                obj_list = []
                for obj in result.statistics.objectives:
                    obj_list.append(-obj)
                result.statistics._objectives = obj_list

            print(">>> FINISH objective:", result.best_state.objective())
        else:
            result = None

        # Result run
        return result

    @staticmethod
    def _is_local_branching(op):
        return (op == DestroyOperators.Local_Branching_05 or
                op == DestroyOperators.Local_Branching_10 or
                op == DestroyOperators.Local_Branching_15 or
                op == DestroyOperators.Local_Branching_20 or
                op == DestroyOperators.Local_Branching_25 or
                op == DestroyOperators.Local_Branching_30 or
                op == DestroyOperators.Local_Branching_35 or
                op == DestroyOperators.Local_Branching_40 or
                op == DestroyOperators.Local_Branching_45 or
                op == DestroyOperators.Local_Branching_50 or
                op == DestroyOperators.Local_Branching_55 or
                op == DestroyOperators.Local_Branching_60 or
                op == DestroyOperators.Local_Branching_65 or
                op == DestroyOperators.Local_Branching_70 or
                op == DestroyOperators.Local_Branching_75 or
                op == DestroyOperators.Local_Branching_80 or
                op == DestroyOperators.Local_Branching_85 or
                op == DestroyOperators.Local_Branching_90 or
                op == DestroyOperators.Local_Branching_95)
    @staticmethod
    def _is_proximity(op):
        return (op == DestroyOperators.Proximity_05 or
                op == DestroyOperators.Proximity_10 or
                op == DestroyOperators.Proximity_15 or
                op == DestroyOperators.Proximity_20 or
                op == DestroyOperators.Proximity_25 or
                op == DestroyOperators.Proximity_30 or
                op == DestroyOperators.Proximity_35 or
                op == DestroyOperators.Proximity_40 or
                op == DestroyOperators.Proximity_45 or
                op == DestroyOperators.Proximity_50 or
                op == DestroyOperators.Proximity_55 or
                op == DestroyOperators.Proximity_60 or
                op == DestroyOperators.Proximity_65 or
                op == DestroyOperators.Proximity_70 or
                op == DestroyOperators.Proximity_75 or
                op == DestroyOperators.Proximity_80 or
                op == DestroyOperators.Proximity_85 or
                op == DestroyOperators.Proximity_90 or
                op == DestroyOperators.Proximity_95)

    def _set_alns_operators(self):

        num_destroy_removed = 0
        # If the problem has no binary, remove Local Branching and Proximity
        if len(self.instance.binary_indexes) == 0:
            for op in self.destroy_ops:
                if self._is_local_branching(op) or self._is_proximity(op):
                    num_destroy_removed += 1
                    continue
                self.alns.add_destroy_operator(op)
        # If the problem has no integer, remove Dins
        elif len(self.instance.integer_indexes) == 0:
            for op in self.destroy_ops:
                if op == DestroyOperators.Dins:
                    num_destroy_removed += 1
                    continue
                self.alns.add_destroy_operator(op)
        else:
            for op in self.destroy_ops:
                self.alns.add_destroy_operator(op)

        for op in self.repair_ops:
            self.alns.add_repair_operator(op)

        num_remaining_destroy = self.selector.num_destroy - num_destroy_removed

        # No more operators left, return failure
        if num_remaining_destroy == 0:
            return False

        # If ops are removed, re-create bandit selector with adjusted arm counter
        if num_destroy_removed > 0:
            if isinstance(self.selector, MABSelector):
                self.selector = MABSelector(scores=self.selector.scores,
                                            num_destroy=num_remaining_destroy,
                                            num_repair=self.selector.num_repair,
                                            learning_policy=self.selector.mab.learning_policy)

        # Ops added successfully
        return True

    @staticmethod
    def _validate_balans_args(destroy_ops, repair_ops, selector, accept, stop, seed, n_jobs, mip_solver):

        # Destroy Type
        for op in destroy_ops:
            check_true(isinstance(op, DestroyType), TypeError("Destroy Type mismatch." + str(op)))

        # Repair Type
        for op in repair_ops:
            check_true(isinstance(op, RepairType), TypeError("Repair Type mismatch." + str(op)))

        # Selector Type
        check_true(isinstance(selector, SelectorType), TypeError("Selector Type mismatch." + str(selector)))

        # Selector Type
        check_true(isinstance(accept, AcceptType), TypeError("Selector Type mismatch." + str(accept)))

        # Stop Type
        check_true(isinstance(stop, StopType), TypeError("Stop Type mismatch." + str(stop)))

        # Seed
        check_true(isinstance(seed, int), TypeError("The seed must be an integer." + str(seed)))

        # Parallel jobs
        check_true(isinstance(n_jobs, int), TypeError("Number of parallel jobs must be an integer." + str(n_jobs)))
        check_true(n_jobs != 0, ValueError("Number of parallel jobs cannot be zero." + str(n_jobs)))

        # MIP solver
        check_true(isinstance(mip_solver, str), TypeError("MIP solver backend must be a string." + str(mip_solver)))
        check_true(mip_solver in ["scip", "gurobi"],
                   ValueError("MIP solver backend must be a scip or gurobi." + str(mip_solver)))

    @staticmethod
    def _validate_solve_args(instance_path):

        check_true(isinstance(instance_path, str), TypeError("Instance path must be a string: " + str(instance_path)))
        check_false(instance_path == "", ValueError("Instance cannot be empty: " + str(instance_path)))
        check_false(instance_path is None, ValueError("Instance cannot be None: " + str(instance_path)))
        check_true(os.path.isfile(instance_path), ValueError("Instance must exist: " + str(instance_path)))
