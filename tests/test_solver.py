import os
from alns.accept import *
from alns.select import *
from alns.stop import *
import numpy as np
from alns.ALNS import ALNS

from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants
from tests.test_base import BaseTest
from balans.base_state import _State
from balans.base_instance import _Instance

from mabwiser.mab import LearningPolicy
import pyscipopt as scip


# TODO: Add back zero objective when figure it out
class SolverTest(BaseTest):

    def test_balans_t1(self):
        # Input
        instance = "model.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Dins,
                       DestroyOperators.Crossover,
                       DestroyOperators.Proximity,
                       DestroyOperators.Mutation2,
                       DestroyOperators.Local_Branching,
                       # DestroyOperators.Zero_Objective,
                       DestroyOperators.Rins,
                       DestroyOperators.Rens]

        repair_ops = [RepairOperators.Repair]

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))

        accept = HillClimbing()
        stop = MaxIterations(1)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        self.assertEqual(result.best_state.objective(), 4)

    def test_balans_t2(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Dins,
                       DestroyOperators.Proximity,
                       DestroyOperators.Mutation2,
                       DestroyOperators.Local_Branching,
                       # DestroyOperators.Zero_Objective,
                       DestroyOperators.Rins,
                       DestroyOperators.Rens,
                       DestroyOperators.Crossover]
        repair_ops = [RepairOperators.Repair]

        instance = _Instance(instance_path)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        # Assert
        self.assertEqual(result.best_state.objective(), -60)

    def test_balans_t3(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Dins,
                       DestroyOperators.Proximity,
                       DestroyOperators.Mutation2,
                       DestroyOperators.Local_Branching,
                       # DestroyOperators.Zero_Objective,
                       DestroyOperators.Rins,
                       DestroyOperators.Rens,
                       DestroyOperators.Crossover]

        model = scip.Model()
        model.hideOutput()
        model.readProblem(instance_path)
        instance = _Instance(model)

        index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, index_to_val, -30)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.RandomState(seed))
        for i in destroy_ops:
            alns.add_destroy_operator(i)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)
        # Retrieve the final solution
        best = result.best_state
        print(f"Best heuristic solution objective is {best.objective()}.")

        self.assertIsBetter(-30, result.best_state.objective(), "minimize")

    def test_balans_t4(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Dins,
                       DestroyOperators.Proximity,
                       DestroyOperators.Mutation2,
                       DestroyOperators.Local_Branching,
                       # DestroyOperators.Zero_Objective,
                       DestroyOperators.Rins,
                       DestroyOperators.Rens,
                       DestroyOperators.Crossover]

        model = scip.Model()
        model.hideOutput()
        model.readProblem(instance_path)
        instance = _Instance(model)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        initial_index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", initial_index_to_val)

        initial2 = _State(instance, initial_index_to_val, -30)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.RandomState(seed))
        for i in destroy_ops:
            alns.add_destroy_operator(i)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)

        # Retrieve the final solution
        best_state = result.best_state
        best_objective = best_state.objective()

        # best_index_to_val = None # best.index_to_val
        #
        # # First variable must remain fixed
        # self.assertEqual(initial_index_to_val[0], best_index_to_val[0])

        print(f"Best heuristic solution objective is {best_objective}.")
        self.assertEqual(best_objective, -60.0)

    def test_balans_t5(self):
        # Input
        instance = "test5.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Dins,
                       DestroyOperators.Proximity,
                       DestroyOperators.Mutation2,
                       DestroyOperators.Local_Branching,
                       # DestroyOperators.Zero_Objective,
                       DestroyOperators.Rins,
                       DestroyOperators.Rens,
                       DestroyOperators.Crossover]

        model = scip.Model()
        model.hideOutput()
        model.readProblem(instance_path)
        instance = _Instance(model)

        index_to_val = {0: 1.0, 1: 0.0, 2: 0.0, 3: 10.0, 4: 10.0, 5: 20.0, 6: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, index_to_val, -40)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.RandomState(seed))
        for i in destroy_ops:
            alns.add_destroy_operator(i)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(5)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)
        # Retrieve the final solution
        best = result.best_state
        print(f"Best heuristic solution objective is {best.objective()}.")
        self.assertEqual(result.best_state.objective(), -60.0)

    def test_balans_t6(self):
        # Input
        instance = "model.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Dins,
                       DestroyOperators.Proximity,
                       DestroyOperators.Mutation2,
                       DestroyOperators.Local_Branching,
                       # DestroyOperators.Zero_Objective,
                       DestroyOperators.Rins,
                       DestroyOperators.Rens,
                       DestroyOperators.Crossover]

        repair_ops = [RepairOperators.Repair]

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))

        accept = HillClimbing()
        stop = MaxIterations(1)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        self.assertEqual(result.best_state.objective(), 4)

    def test_balans_t7(self):
        # Input
        instance = "test2.5.cip"
        instance_path = os.path.join(Constants.DATA_TOY, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Rins,
                       DestroyOperators.Local_Branching]

        repair_ops = [RepairOperators.Repair]

        # for destroy_op in destroy_ops:
        if repair_ops:

            initial_index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}

            selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                                   learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
            accept = AlwaysAccept()
            stop = MaxIterations(1)
            seed = 123456
            balans = Balans([DestroyOperators.Rins], repair_ops, selector, accept, stop, seed)
            # Run
            result = balans.solve(instance_path, initial_index_to_val)
            # Retrieve the final solution
            best_state = result.best_state

            print("Best solution first loop:", result.best_state.solution())

            best_sol_init = result.best_state.solution()

            couple_ops = [DestroyOperators.Mutation2]

            # Given best_sol_init run one iteration with other op
            for second_op in destroy_ops:
                if not second_op == DestroyOperators.Mutation2:
                    couple_ops.append(second_op)

                    # Selector
                    selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                                           learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
                    accept = AlwaysAccept()
                    stop = MaxIterations(1)
                    seed = 123456
                    # Solver
                    balans = Balans([DestroyOperators.Local_Branching], repair_ops, selector, accept, stop, seed)
                    initial_index_to_val = best_sol_init
                    print("initial index to val:", initial_index_to_val)

                    # Run
                    result = balans.solve(instance_path, initial_index_to_val)

                    # Retrieve the final solution
                    if result:
                        best_state = result.best_state
                        best_solution = best_state.solution()

                        print("Best solution second loop:", result.best_state.solution())

                    accept = AlwaysAccept()
                    stop = MaxIterations(2)
                    couple_ops = [DestroyOperators.Rins, DestroyOperators.Local_Branching]

                    selector = RandomSelect(num_destroy=2, num_repair=1)
                    # Solver
                    balans = Balans(couple_ops, repair_ops, selector, accept, stop, seed)
                    # Run
                    initial_index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}

                    result2 = balans.solve(instance_path, initial_index_to_val)

                    if result2:
                        # Retrieve the final solution
                        best_state2 = result2.best_state
                        best_objective2 = best_state2.objective()
                        best_solution2 = best_state2.solution()

                        print("Best solution final:", result2.best_state.solution())
                        print("Best solution final value:", result2.best_state.objective())

                        self.assertDictEqual(best_solution2, best_solution)
