from tests.test_base import BaseTest
import gurobipy as grb
from gurobi_onboarder import init_gurobi

class UtilsTest(BaseTest):

    def test_basic_gurobi(self):

        gurobi_venv, GUROBI_FOUND = init_gurobi.initialize_gurobi()

        # Create a new model
        m = grb.Model(env=gurobi_venv)
        m.setParam("OutputFlag", 0)

        # Create variables
        x = m.addVar(vtype='I', name="x")
        y = m.addVar(vtype='I', name="y")
        z = m.addVar(vtype='B', name="z")

        # Set objective function
        m.setObjective(x + y + 2 * z, grb.GRB.MAXIMIZE)

        # Add constraints
        m.addConstr(x + 2 * y + 3 * z <= 4)
        m.addConstr(x + y >= 1)

        # Solve it!
        m.optimize()

        print(f"Optimal objective value: {m.objVal}")
        print(f"Solution values: x={x.X}, y={y.X}, z={z.X}")

        self.assertTrue(True)