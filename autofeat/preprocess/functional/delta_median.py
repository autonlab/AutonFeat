import numpy as np
from typing import Union, Callable
from autofeat.functional import median_tf
from autofeat.preprocess.functional import delta_tf


def delta_median_tf(x: np.ndarray, method: str = 'linear', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
    """
    Preprocess the signal `x` by shifting each element of `x` by the median of `x`.

    Args:
        x: The array to shift by its median.

        method: The method to use when computing the quantile. Default is 'linear'. See `numpy.quantile` for more information.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The shifted signal.
    """
    median = median_tf(x, method=method, where=where)
    return delta_tf(x, delta=median, where=where)
