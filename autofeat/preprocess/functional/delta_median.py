import numpy as np
from typing import Union, Callable
from autofeat.functional import median_tf
from autofeat.preprocess.functional import delta_tf


def delta_median_tf(x: np.ndarray, method: str = 'linear', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the difference between the values in `x` and `median` where `where` is `True`.

    Args:
        `x`: The array to compute the delta with.

        `method`: The method to use when computing the quantile. Default is 'linear'. See `numpy.quantile` for more information.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The difference between the of the values in `x` and `median` where `where` is `True`.

    """
    median = median_tf(x, method=method, where=where)
    return delta_tf(x, delta=median, where=where)
