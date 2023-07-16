import random
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tests.test_base import BaseTest



def dinst(state: State, rnd_state):

    return State(state.x, state.model)
