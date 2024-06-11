import os
from typing import List, Optional, Dict
from typing import NamedTuple

import numpy as np
from alns.ALNS import ALNS
from alns.Result import Result
from alns.accept import MovingAverageThreshold, GreatDeluge, HillClimbing
from alns.accept import LateAcceptanceHillClimbing, NonLinearGreatDeluge, AlwaysAccept
from alns.accept import RecordToRecordTravel, SimulatedAnnealing, RandomAccept
from alns.select import AlphaUCB, MABSelector, RandomSelect, RouletteWheel, SegmentedRouletteWheel
from alns.stop import MaxIterations, MaxRuntime, NoImprovement, StoppingCriterion

from balans.base_instance import _Instance
from balans.base_state import _State
from balans.destroy.crossover import crossover
from balans.destroy.dins import dins
from balans.destroy.local_branching import local_branching_50
from balans.destroy.mutation import mutation_25, mutation_50, mutation_75, mutation_binary_50
from balans.destroy.proximity import proximity
from balans.destroy.rens import rens_50
from balans.destroy.rins import rins, rins_random_50
from balans.destroy.zero_objective import zero_objective
from balans.repair.repair import repair
from balans.utils import Constants, check_false, check_true, create_rng


class DestroyOperators(NamedTuple):
    Crossover = crossover
    Dins = dins
    Local_Branching = local_branching_50
    Mutation = mutation_25
    Mutation2 = mutation_50
    Mutation3 = mutation_75
    Mutation_Binary = mutation_binary_50
    Proximity = proximity
    Rens = rens_50
    Rins = rins
    Rins_Random = rins_random_50
    Zero_Objective = zero_objective


class RepairOperators(NamedTuple):
    Repair = repair


# Type Declarations
DestroyType = (type(DestroyOperators.Crossover),
               type(DestroyOperators.Dins),
               # type(DestroyOperators.Dins_Random),
               type(DestroyOperators.Local_Branching),
               type(DestroyOperators.Mutation),
               type(DestroyOperators.Mutation2),
               type(DestroyOperators.Mutation3),
               type(DestroyOperators.Proximity),
               type(DestroyOperators.Rens),
               type(DestroyOperators.Rins),
               type(DestroyOperators.Zero_Objective))

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

    def __init__(self,
                 destroy_ops: List,
                 repair_ops: List,
                 selector: SelectorType,
                 accept: AcceptType,
                 stop: StopType,
                 seed: int = Constants.default_seed,  # The random seed
                 n_jobs: int = 1,  # Number of parallel jobs
                 backend: str = None  # Parallel backend implementation
                 ):

        # Validate arguments
        self._validate_balans_args(destroy_ops, repair_ops, selector, accept, stop, seed, n_jobs, backend)

        # Parameters
        self.destroy_ops = destroy_ops
        self.repair_ops = repair_ops
        self.selector = selector
        self.accept = accept
        self.stop = stop
        self.seed = seed
        self.n_jobs = n_jobs
        self.backend = backend

        # RNG
        self._rng = create_rng(self.seed)
        self.alns_seed = self._rng.randint(0, self.seed)

        # ALNS
        self.alns = ALNS(np.random.RandomState(self.alns_seed))
        for op in destroy_ops:
            self.alns.add_destroy_operator(op)
        for op in repair_ops:
            self.alns.add_repair_operator(op)

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
        """
        self._validate_solve_args(instance_path)

        # Create instance from mip file
        self._instance = _Instance(instance_path)

        # Initial solution
        self._initial_index_to_val, self._initial_obj_val = self._instance.solve(is_initial_solve=True,
                                                                                 index_to_val=index_to_val)
        print(">>> START objective:", self._initial_obj_val)
        print(">>> START objective index:", self._initial_index_to_val)

        # Initial state and solution
        initial_state = _State(self._instance, self.initial_index_to_val,
                               self.initial_obj_val, previous_index_to_val=self._initial_index_to_val)

        result = self.alns.iterate(initial_state, self.selector, self.accept, self.stop)

        print(">>> FINISH objective:", result.best_state.objective())

        # Result run
        return result

    @staticmethod
    def _validate_balans_args(destroy_ops, repair_ops, selector, accept, stop, seed, n_jobs, backend):

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
        if backend is not None:
            check_true(isinstance(backend, str), TypeError("Parallel backend must be a string." + str(backend)))

    @staticmethod
    def _validate_solve_args(instance_path):

        check_true(isinstance(instance_path, str), TypeError("Instance path must be a string" + str(instance_path)))
        check_false(instance_path == "", ValueError("Instance cannot be empty" + str(instance_path)))
        check_false(instance_path is None, ValueError("Instance cannot be None" + str(instance_path)))
        check_true(os.path.isfile(instance_path), ValueError("Instance must exist" + str(instance_path)))

        # if gap is not None:
        #     check_true(isinstance(gap, int), TypeError("Gap must be an integer." + str(gap)))
        #     check_true(gap >= 0, ValueError("Gap must be non-negative" + str(gap)))
        #
        # if time is not None:
        #     check_true(isinstance(time, int), TypeError("Time must be an integer." + str(time)))
        #     check_true(time >= 0, ValueError("Time must be non-negative" + str(gap)))



# class DestroyOperators(NamedTuple):
#     class Crossover(NamedTuple):
#         """Crossover Operator
#         This operator..
#         """
#         pass
#
#     class Dins(NamedTuple):
#         """ DINS Operator
#         This operator ..
#
#         is_random: bool
#             Whether the destroy is random
#
#         delta: Num
#            If int, represents the number of variables to destroy
#            If float, represents the percentage of variables to destroy (0.0, 1.0]
#         """
#         is_random: bool = False
#         delta: Num = 0.25
#
#         def _validate(self):
#             check_true(isinstance(self.is_random, bool), TypeError("Is_random must be an boolean."))
#             check_true(isinstance(self.delta, (int, float)), TypeError("Delta must be an integer or float."))
#             check_true(0 <= self.delta, ValueError("Delta must be between positive."))
#
#     class LocalBranching(NamedTuple):
#         """ Local Branching Operator
#         This operator ..
#
#         delta: Num
#            If int, represents the number of variables to destroy
#            If float, represents the percentage of variables to destroy (0.0, 1.0]
#         """
#         delta: Num = 0.25
#
#         def _validate(self):
#             check_true(isinstance(self.delta, (int, float)), TypeError("Delta must be an integer or float."))
#             check_true(0 <= self.delta, ValueError("Delta must be between positive."))
#
#         class Mutation(NamedTuple):
#             """ Mutation Operator
#             This operator ..
#
#             is_binary: bool
#                 Whether the destroy operates on binary variables only
#
#             delta: Num
#                If int, represents the number of variables to destroy
#                If float, represents the percentage of variables to destroy (0.0, 1.0]
#             """
#             is_binary: bool = False
#             delta: Num = 0.25
#
#             def _validate(self):
#                 check_true(isinstance(self.is_binary, bool), TypeError("Is_binary must be an boolean."))
#                 check_true(isinstance(self.delta, (int, float)), TypeError("Delta must be an integer or float."))
#                 check_true(0 <= self.delta, ValueError("Delta must be between positive."))
