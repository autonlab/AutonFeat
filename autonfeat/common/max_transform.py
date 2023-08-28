# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Callable, Union

from autonfeat.core import Transform
from autonfeat.functional import max_tf


class MaxTransform(Transform):
    """
    Compute the max of the values in `x`.
    """
    # Dunder methods
    def __init__(self, name: str = "Max") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x), initial: Union[int, float, np.int_, np.float_] = -np.inf) -> Union[np.float_, np.int_]:
        """
        Compute the max of the signal window provided.

        Args:
            signal_window: The signal window to find the max of.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

            initial: The initial value to use when computing the max. Default is `-np.inf`.

        Returns:
            A scalar value representing the max of the signal.
        """
        return max_tf(signal_window, where=where, initial=initial)
