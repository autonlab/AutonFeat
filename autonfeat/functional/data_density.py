# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Union, Callable
from autonfeat.functional import n_valid_tf


def data_density_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the data density of the array `x`.

    Args:
        x: The array to compute the data density of.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The data density of measurements in `x`.

    Raises:
        DivideByZeroError: If `x` is empty.
    """
    size = x.shape[0]
    if size == 0:
        raise ZeroDivisionError("Cannot compute data density of empty array.")

    return n_valid_tf(x, where=where) / size
