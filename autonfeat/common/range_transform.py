# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Union, Callable

from autonfeat.core import Transform
from autonfeat.functional import range_tf


class RangeTransform(Transform):
    """
    Compute the range of the values.
    """
    # Dunder methods
    def __init__(self, name: str = "Range") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[np.float_, np.int_]:
        """
        Compute the range of the values in `x`.

        Args:
            signal_window: The array to compute the range of.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            A scalar value representing the range of the signal.
        """
        return range_tf(signal_window, where=where)
