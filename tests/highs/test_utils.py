import os

from highspy import Highs, HighsModel, HighsModelStatus, ObjSense

from balans.utils import Constants
from tests.test_base import BaseTest


class UtilsTest(BaseTest):

    BaseTest.mip_solver = Constants.highs_solver

    def test_basic_highs(self):

        # Create HiGHS model
        instance = "model.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        model = Highs()
        model.readModel(instance_path)

        # Create a HiGHS model
        model = Highs()
        model.setOptionValue('random_seed', 42)

        x1 = model.addVariable(lb=-model.inf)
        x2 = model.addVariable(lb=-model.inf)

        model.addConstrs(x2 - x1 >= 2,
                         x1 + x2 >= 0)

        model.maximize()
        model.changeObjectiveSense(ObjSense.kMinimize)

        # Solve the problem
        model.run()

        vars = model.getVariablws()
        print("vars:", vars)
        print("vars:", vars[0].index)
        print("vars:", vars[0])

        # org_objective_fn = model.setMaximize()
        # org_objective_sense = model.getObjectiveSense()

        # Get the solution
        solution = model.getSolution()
        print("Optimal solution:", list(solution.col_value))
        print("Optimal objective value:", model.getObjectiveValue())

        # Get the solution information
        info = model.getInfo()

        status = model.getModelStatus()
        self.assertEqual(status, HighsModelStatus.kOptimal)

        # Retrieve the objective function value
        objective_value = info.objective_function_value
        print("Optimal objective value:", objective_value)

        self.assertTrue(True)