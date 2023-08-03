import pyscipopt
from pyscipopt import Model
from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_PARAMSETTING, SCIP_HEURTIMING

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


model.addCons(x2 + x3  + x5 + x6 == 60)


#max problem = 32, when x3=0
#for exmaple x1=20, x3=20, x4=20


#model.writeParams('default.set', onlychanged=False)

model.writeProblem("test.7.0.cip")

#model.hideOutput()
model.optimize()

