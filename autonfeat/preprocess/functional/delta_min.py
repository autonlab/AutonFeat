# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Union, Callable
from autonfeat.functional import min_tf
from autonfeat.preprocess.functional import delta_tf


def delta_min_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x), initial: Union[int, float, np.int_, np.float_] = np.inf) -> np.ndarray:
    """
    Preprocess the signal `x` by shifting each element of `x` by the minimum of `x`.

    Args:
        x: The array to shift by its minimum.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        initial: The initial value for the minimum. Default is `np.inf`.

    Returns:
        The shifted signal.
    """
    min = min_tf(x, where=where, initial=initial)
    return delta_tf(x, delta=min, where=where)
