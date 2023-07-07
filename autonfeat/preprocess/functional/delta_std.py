import numpy as np
from typing import Union, Callable
from autonfeat.functional import std_tf
from autonfeat.preprocess.functional import delta_tf


def delta_std_tf(x: np.ndarray, ddof: Union[int, np.int_] = 0, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
    """
    Preprocess the signal `x` by shifting each element of `x` by the standard deviation of `x`.

    Args:
        x: The array to shift by its standard deviation.

        ddof: The delta degrees of freedom. Default is `0`. See `numpy.std` for more information.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The shifted signal.
    """
    std = std_tf(x, ddof=ddof, where=where)
    return delta_tf(x, delta=std, where=where)
