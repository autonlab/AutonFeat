# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Union, Callable
from autonfeat.core import Preprocess
from autonfeat.preprocess.functional import delta_median_tf


class DeltaMedianPreprocessor(Preprocess):
    """
    Preprocess the signal by shifting the signal up by the median value.
    """
    # Dunder methods
    def __init__(self, name: str = "Delta Median") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, method: str = 'linear', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
        """
        Compute the difference between the values in `signal` and `median` where `where` is `True`.

        Args:
            signal: The array to compute the delta with.

            method: The method to use when computing the quantile. Default is 'linear'. See `numpy.quantile` for more information.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The shifted signal.
        """
        return delta_median_tf(x=signal, method=method, where=where)
