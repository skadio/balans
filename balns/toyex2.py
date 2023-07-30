from pyscipopt import Model
import pandas as pd
import numpy as np


model = Model()


x0 = model.addVar(vtype="B")
x1 = model.addVar(vtype="B")
x2 = model.addVar(vtype="I")
x3 = model.addVar(vtype="I")
x4 = model.addVar(vtype="I")
x5 = model.addVar(vtype="C")
x6 = model.addVar(vtype="C")


model.addCons(x2 + x3 + x4 + x5 + x6 == 60)

model.addCons(abs(x0) == 2)
#max problem = 32, when x3=0
#for exmaple x1=20, x3=20, x4=20

# Set objective function
model.setObjective(-x2 -x3 - x5-x0, "minimize")


model.hideOutput()
model.optimize()



model.writeProblem("test5.7.cip")
