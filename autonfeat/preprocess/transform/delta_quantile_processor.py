# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Union, Callable
from autonfeat.core import Preprocess
from autonfeat.preprocess.functional import delta_quantile_tf


class DeltaQuantilePreprocessor(Preprocess):
    """
    Preprocess the signal by shifting each element in the signal by the quantile of the signal.
    """
    # Dunder methods
    def __init__(self, name: str = "Delta Quantile") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, q: Union[float, np.float_], method: str = 'linear', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
        """
        Compute the quantile of the signal and shift the signal by this quantile.

        Args:
            signal: The array to compute the delta with.

            q: The quantile to compute. Must be between 0 and 1.

            method: The method to use when computing the quantile. Default is 'linear'. See `numpy.quantile` for more information.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The shifted signal.
        """
        return delta_quantile_tf(x=signal, q=q, method=method, where=where)
