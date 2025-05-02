import pyscipopt as scip

from balans.utils import Constants
from tests.test_base import BaseTest


class UtilsTest(BaseTest):

    BaseTest.mip_solver = Constants.scip_solver

    def test_basic_scip(self):
        model = scip.Model("puzzle")
        x = model.addVar(vtype="I", name="octopusses")
        y = model.addVar(vtype="I", name="turtles")
        z = model.addVar(vtype="I", name="cranes")

        # Set up constraint for number of heads
        model.addCons(x + y + z == 32, name="Heads")

        # Set up constraint for number of legs
        model.addCons(8 * x + 4 * y + 2 * z == 80, name="Legs")

        # Set objective function
        model.setObjective(x + y, "minimize")

        model.hideOutput()
        model.optimize()

        solution = model.getBestSol()

        print("Optimal value:", model.getObjVal())
        print((x.name, y.name, z.name), " = ", (model.getVal(x), model.getVal(y), model.getVal(z)))

        self.assertTrue(True)