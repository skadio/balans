from pyscipopt import Model
import pandas as pd
import numpy as np

SEED = 42
np.random.seed(SEED)
rnd_state = np.random.RandomState(SEED)


def solve(first=None):
    if first:
        # Set objective function
        model = Model("puzzle")
        model.hideOutput()
        x = model.addVar(vtype="I", name="octopusses")
        y = model.addVar(vtype="I", name="turtles")
        z = model.addVar(vtype="I", name="cranes")

        # Set up constraint for number of heads
        # model.addCons(x + y + z == 32, name="Heads")

        # Set up constraint for number of legs
        model.addCons(8 * x + 4 * y + 2 * z == 80, name="Legs")
        model.setObjective(x + y, "minimize")
        model.addCons(x + y + z == 32, name="Heads")

        model.optimize()
        solution = model.getBestSol()

        sol = model.createPartialSol()

        for var in model.getVars()[:-1]:
            # sol[var] = solution[var.getIndex()]
            print("Varss", var)

        print(solution, "solution 1")
        print("Optimal value 1:", model.getObjVal())
        print((x.name, y.name, z.name), " = ", (model.getVal(x), model.getVal(y), model.getVal(z)))

        model.freeTransform()


    else:
        # Set objective function
        model = Model("puzzle")
        model.hideOutput()
        x = model.addVar(vtype="I", name="octopusses")
        y = model.addVar(vtype="I", name="turtles")
        z = model.addVar(vtype="I", name="cranes")

        # Set up constraint for number of heads
        # model.addCons(x + y + z == 32, name="Heads")

        # Set up constraint for number of legs
        model.addCons(8 * x + 4 * y + 2 * z == 80, name="Legs")
        model.addCons(x + y + z == 32, name="Heads")

        for var in model.getVars():
            if var.getIndex() == 1:
                model.addCons(var == 8)
                print(var, "cemal", var.name)

        model.setObjective(x + y, "minimize")
        model.optimize()
        solution = model.getBestSol()

        print("Optimal value conssss:", model.getObjVal())
        print((x.name, y.name, z.name), " = ", (model.getVal(x), model.getVal(y), model.getVal(z)))
        model.freeTransform()


model = Model("puzzle")
x = model.addVar(vtype="I", name="octopusses")
y = model.addVar(vtype="I", name="turtles")
z = model.addVar(vtype="I", name="cranes")

# Set up constraint for number of heads
model.addCons(x + y + z == 32, name="Heads")

# Set up constraint for number of legs
model.addCons(8*x + 4*y + 2*z == 80, name="Legs")

# Set objective function
model.setObjective(x + y, "minimize")

model.hideOutput()
model.optimize()

#solution = model.getBestSol()

print("Optimal value:", model.getObjVal())
print((x.name, y.name, z.name), " = ", (model.getVal(x), model.getVal(y), model.getVal(z)))
