# -*- coding: utf-8 -*-

import random
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tests.test_base import BaseTest


class ExampleTest(BaseTest):

    def test_tautology(self):

        self.assertEqual(True, True)