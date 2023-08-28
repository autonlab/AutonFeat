# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Union, Callable
import warnings


def min_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x), initial: Union[int, float, np.int_, np.float_] = np.inf) -> Union[float, np.float_]:
    """
    Compute the min of the values in `x` where `where` is True.

    Args:
        x: The array to compute the min of.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        initial: The initial value to use when computing the min. Default is `np.inf`.

    Returns:
        The min of the values in `x` where `where` is True.
    """
    # Vectorize where fn
    where_fn = np.vectorize(pyfunc=where)
    if x.dtype != np.float_ and initial.dtype == np.float_:
        warnings.warn("Only float arrays can be used with an initial comparator of type float, but the input array has dtype {}. Trying to cast array elements to numpy float.".format(x.dtype))
        x = x.astype(np.float_)
    return np.amin(x, axis=0, where=where_fn(x), initial=initial)
