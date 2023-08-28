# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Union, Callable
from autonfeat.functional import quantile_tf


def iqr_tf(x: np.ndarray, method: str = 'linear', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the inter-quartile range of the values in `x`.

    Args:
        x: The array to compute the IQR of.

        method: The method to use when computing the quantiles. Default is 'linear'. See `numpy.quantile` for more information.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The IQR of the values in `x`.
    """
    return quantile_tf(x, q=0.75, method=method, where=where) - quantile_tf(x, q=0.25, method=method, where=where)
