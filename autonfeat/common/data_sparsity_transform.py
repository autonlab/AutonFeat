# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Callable, Union

from autonfeat.core import Transform
from autonfeat.functional import data_sparsity_tf


class DataSparsityTransform(Transform):
    """
    Compute the data sparsity of a signal window `x`.
    """
    # Dunder methods
    def __init__(self, name: str = "Data Sparsity") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[np.float_, np.int_]:
        """
        Compute the data sparsity of the array `x`.

        Args:
            signal_window: The signal window to find the data sparsity of.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            A scalar value representing the data sparsity of `x`.
        """
        return data_sparsity_tf(x=signal_window, where=where)
