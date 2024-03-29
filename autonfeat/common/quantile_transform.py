# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Union, Callable

from autonfeat.core import Transform
from autonfeat.functional import quantile_tf


class QuantileTransform(Transform):
    """
    Compute the q-th quantile of the values.
    """
    # Dunder methods
    def __init__(self, name: str = "Quantile") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, q: Union[float, np.float_], method: str = 'linear', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[np.float_, np.int_]:
        """
        Compute the q-th quantile of the values in `x`.

        Args:
            signal_window: The array to compute the q-th quantile of.

            q: The quantile to compute. `q` belongs to [0, 1].

            method: The method to use when computing the quantile. Default is 'linear'. See `numpy.quantile` for more information.

            where: `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            A scalar value representing the q-th quantile of the signal.
        """
        return quantile_tf(signal_window, q=q, method=method, where=where)
