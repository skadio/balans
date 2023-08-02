from pyscipopt import Model
import pandas as pd
import numpy as np


model = Model()


x0 = model.addVar(vtype="B")
x1 = model.addVar(vtype="B")
x2 = model.addVar(vtype="I")
x3 = model.addVar(vtype="I")
x4 = model.addVar(vtype="B")
x5 = model.addVar(vtype="C")
x6 = model.addVar(vtype="C")


model.addCons(x2 + x3  + x5 + x6 == 60)


#max problem = 32, when x3=0
#for exmaple x1=20, x3=20, x4=20

# Set objective function
model.setObjective(-x0-x1-x2 -x3 -x4- x5, "minimize")


model.hideOutput()
model.optimize()



model.writeProblem("test5.13.cip")
