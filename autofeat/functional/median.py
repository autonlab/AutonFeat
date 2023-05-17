import numpy as np
from typing import Union, Callable
from autofeat.functional import quantile_tf


def median_tf(x: np.ndarray, method: str = 'linear', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the median of the values in `x`.

    Args:
        `x`: The array to compute the median of.

        `method`: The method to use when computing the quantile. Default is 'linear'. See `numpy.quantile` for more information.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The median of the values in `x`.

    """
    return quantile_tf(x, q=0.5, method=method, where=where)
