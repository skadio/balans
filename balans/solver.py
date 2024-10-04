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
from balans.destroy.local_branching import local_branching_10, local_branching_25, local_branching_50
from balans.destroy.local_branching_relax import local_branching_relax_10, local_branching_relax_25
from balans.destroy.mutation import mutation_25, mutation_50, mutation_75
from balans.destroy.proximity import proximity_05, proximity_15, proximity_30
from balans.destroy.random_objective import random_objective
from balans.destroy.rens import rens_25, rens_50, rens_75
from balans.destroy.rins import rins_25, rins_50, rins_75
from balans.repair.repair import repair
from balans.utils import Constants, check_false, check_true, create_rng


class DestroyOperators(NamedTuple):
    Crossover = crossover
    Dins = dins
    Local_Branching_10 = local_branching_10
    Local_Branching_25 = local_branching_25
    Local_Branching_50 = local_branching_50
    Local_Branching_Relax_10 = local_branching_relax_10
    Local_Branching_Relax_25 = local_branching_relax_25
    Mutation_25 = mutation_25
    Mutation_50 = mutation_50
    Mutation_75 = mutation_75
    Proximity_05 = proximity_05
    Proximity_15 = proximity_15
    Proximity_30 = proximity_30
    Rens_25 = rens_25
    Rens_50 = rens_50
    Rens_75 = rens_75
    Rins_25 = rins_25
    Rins_50 = rins_50
    Rins_75 = rins_75
    Random_Objective = random_objective


class RepairOperators(NamedTuple):
    Repair = repair


# Type Declarations
DestroyType = (type(DestroyOperators.Crossover),
               type(DestroyOperators.Dins),
               type(DestroyOperators.Local_Branching_10),
               type(DestroyOperators.Local_Branching_25),
               type(DestroyOperators.Local_Branching_50),
               type(DestroyOperators.Local_Branching_Relax_10),
               type(DestroyOperators.Local_Branching_Relax_25),
               type(DestroyOperators.Mutation_25),
               type(DestroyOperators.Mutation_50),
               type(DestroyOperators.Mutation_75),
               type(DestroyOperators.Proximity_05),
               type(DestroyOperators.Proximity_15),
               type(DestroyOperators.Proximity_30),
               type(DestroyOperators.Rens_25),
               type(DestroyOperators.Rens_50),
               type(DestroyOperators.Rens_75),
               type(DestroyOperators.Rins_25),
               type(DestroyOperators.Rins_50),
               type(DestroyOperators.Rins_75),
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
                 mip_solver: str = "scip"  # MIP solver scip/gurobi
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
        return (op == DestroyOperators.Local_Branching_10 or
                op == DestroyOperators.Local_Branching_25 or
                op == DestroyOperators.Local_Branching_50)

    @staticmethod
    def _is_proximity(op):
        return (op == DestroyOperators.Proximity_05 or
                op == DestroyOperators.Proximity_15 or
                op == DestroyOperators.Proximity_30)

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
