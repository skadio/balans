import os
import numpy as np
from typing import List, Optional, Dict

from alns.ALNS import ALNS
from alns.Result import Result

from balans.base_instance import _Instance
from balans.base_state import _State
from balans.utils import Constants, check_false, check_true, create_rng

from balans.destroy import DestroyOperators
from balans.repair import RepairOperators

from alns.accept import AdaptiveThreshold, GreatDeluge, HillClimbing
from alns.accept import LateAcceptanceHillClimbing, NonLinearGreatDeluge, RandomWalk
from alns.accept import RecordToRecordTravel, SimulatedAnnealing, WorseAccept
from alns.select import AlphaUCB, MABSelector, RandomSelect, RouletteWheel, SegmentedRouletteWheel
from alns.stop import MaxIterations, MaxRuntime, NoImprovement, StoppingCriterion

# Type Declarations
DestroyType = (type(DestroyOperators.Crossover),
               type(DestroyOperators.Dins),
               type(DestroyOperators.Local_Branching),
               type(DestroyOperators.Local_Branching2),
               type(DestroyOperators.Local_Branching3),
               type(DestroyOperators.Mutation),
               type(DestroyOperators.Mutation2),
               type(DestroyOperators.Mutation3),
               type(DestroyOperators.Mutation4),
               type(DestroyOperators.No_Objective),
               type(DestroyOperators.Proximity),
               type(DestroyOperators.Rens),
               type(DestroyOperators.Rins))

RepairType = (type(RepairOperators.Repair))

AcceptType = (AdaptiveThreshold,
              GreatDeluge,
              HillClimbing,
              LateAcceptanceHillClimbing,
              NonLinearGreatDeluge,
              RandomWalk,
              RecordToRecordTravel,
              SimulatedAnnealing,
              WorseAccept)

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

        # ALNS
        alns_seed = self._rng.randint(0, Constants.default_seed)
        self.alns = ALNS(np.random.RandomState(alns_seed))
        for op in destroy_ops:
            self.alns.add_destroy_operator(op)
        for op in repair_ops:
            self.alns.add_repair_operator(op)

        # Instance and the first solution
        self._instance: Optional[_Instance] = None
        self._initial_var_to_val: Optional[Dict[int, float]] = None
        self._initial_obj_val: Optional[float] = None

    @property
    def instance(self) -> _Instance:
        return self._instance

    @property
    def initial_var_to_val(self) -> Dict[int, float]:
        return self._initial_var_to_val

    @property
    def initial_obj_val(self) -> float:
        return self._initial_obj_val

    def solve(self, instance_path) -> Result:

        self._validate_solve_args(instance_path)

        # Create instance from mip file
        self._instance = _Instance(instance_path)

        # Initial solution
        self._initial_var_to_val, self._initial_obj_val, lp_var_to_val, lp_obj_val \
            = self._instance.solve(is_initial_solve=True)
        print(">>> START objective:", self._initial_obj_val)
        # LP solution
        # self._lp_var_to_val, self._lp_obj_val = self._instance.lp_solve()
        print(">>> LP objective:", lp_obj_val)

        # Initial state and solution
        initial_state = _State(self._instance, self.initial_var_to_val, self.initial_obj_val,
                               lp_var_to_val=lp_var_to_val, lp_obj_val=lp_obj_val)

        print()
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
