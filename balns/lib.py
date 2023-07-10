#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
:Author: xxx
:Version: 1.0.0 of February 29, 2020

This module defines the public interface of the **XXX Library** 
providing access to the following modules:

    - ``XX``
"""

from typing import List, Union, Dict, NamedTuple, NoReturn, Callable, Optional
from lib.utils import Num, check_true, Constants

import numpy as np
import pandas as pd

__author__ = "xxx"
__copyright__ = "Copyright (C), xxx"


class SomeCommonFunctionality(NamedTuple):
    class ApproachA(NamedTuple):
        """ApproachA

        This approachA

        Attributes
        ----------
        param: Num
            param
            Default value is 0.05.

        Example
        -------
            >>> from xx.xx import xx
            >>> xx.run()
            'output'
        """
        param: Num = 0.05

        def _validate(self):
            check_true(isinstance(self.param, (int, float)), TypeError("param must be an integer or float."))

    class ApproachB(NamedTuple):
        """ ApproachB

        Attributes
        ----------
        param: Num
            param
            Default value is 1.0.

        Example
        -------
            >>> from xx.xx import xx
            >>> xx.run()
            'output'
        """
        param: Num = 1.0

        def _validate(self):
            check_true(0 < self.param, ValueError("The value of param must be greater than zero."))


class Library:
    """**Library: Some Library**

    XXX is a research library for

    Attributes
    ----------
    param : list

    Example
    -------
        >>> from xx.xx import xx
        >>> xx.run()
        'output'
    """

    def __init__(self, seed: int = Constants.default_seed):
        """Initializes xxx given the arguments.

        Validates the arguments and raises exception in case there are violations.

        Parameters
        ----------
        seed : numbers.Rational, optional
            The random seed to initialize the random number generator.
            Default value is set to Constants.default_seed.value

        Raises
        ------
        TypeError:  Seed is not an integer.

        ValueError: Invalid seed value.
        """

        # Validate arguments
        Library._validate_args(seed)

        # Save the arguments
        self.seed = seed

        # Create the random number generator
        self._rng = np.random.RandomState(seed=self.seed)

    @staticmethod
    def _validate_args(seed) -> NoReturn:
        """
        Validates arguments for the constructor.
        """

        # Seed
        check_true(isinstance(seed, int), TypeError("The seed must be an integer."))
