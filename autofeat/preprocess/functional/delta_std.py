import numpy as np
from typing import Union, Callable
from autofeat.functional import std_tf
from autofeat.preprocess.functional import delta_tf


def delta_std_tf(x: np.ndarray, ddof: Union[int, np.int_] = 0, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
    """
    Compute the difference between the values in `x` and `std` where `where` is `True`.

    Args:
        `x`: The array to compute the delta with.

        `ddof`: The delta degrees of freedom. Default is `0`. See `numpy.std` for more information.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The difference between the of the values in `x` and `std` where `where` is `True`.

    """
    std = std_tf(x, ddof=ddof, where=where)
    return delta_tf(x, delta=std, where=where)
