from pyscipopt import Model
import pandas as pd
import numpy as np
import os
from alns.accept import *
from alns.select import *
from alns.stop import *
from pyscipopt import Model
import numpy as np

from balans.destroy import DestroyOperators
from balans.repair import RepairOperators
from balans.solver import Balans
from balans.utils import Constants
from tests.test_base import BaseTest
from balans.base_state import _State
from balans.base_instance import _Instance

from mabwiser.mab import LearningPolicy, NeighborhoodPolicy

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = TEST_DIR + os.sep + ".." + os.sep
SEED = 42
np.random.seed(SEED)
rnd_state = np.random.RandomState(SEED)

SEED = 42
np.random.seed(SEED)
rnd_state = np.random.RandomState(SEED)

instance = "markshare_4_0.mps.gz"
instance_path = os.path.join(ROOT_DIR, Constants.DATA_DIR, instance)
model = Model()
model.hideOutput()
model.readProblem(instance_path)

#model.optimize()

# solution = model.getBestSol()

#print("Optimal value:", model.getObjVal())

# model.writeProblem("model4.cip")

model.writeProblem("market4.0.cip")

variables = model.getVars()
# Set discrete indexes MODIFIED
discrete = []
for var in variables:
    if var.vtype() == 'CONTINUOUS':
        discrete.append(var.getIndex())
print("disc", discrete)

destroy_size = int(0.5 * len(discrete))

destroy_set = set(rnd_state.choice(discrete, destroy_size))
print("disc set", destroy_set)
# Set discrete indexes MODIFIED
discrete1 = []
for var in variables:
    if var.vtype() == 'INTEGER' or var.vtype() == 'BINARY':
        discrete1.append(var.getIndex())
print("disc1", discrete1)
#model.writeProblem("model5.cip")
