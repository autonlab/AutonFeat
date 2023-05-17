import numpy as np
from typing import Union, Callable
from autofeat.functional import max_tf, delta_tf


def delta_max_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x), initial: Union[int, float, np.int_, np.float_] = -np.inf) -> Union[float, np.float_]:
    """
    Compute the difference between the values in `x` and `max` where `where` is `True`.

    Args:
        `x`: The array to compute the delta with.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        `initial`: The initial value for the maximum. Default is `-np.inf`.

    Returns:
        The difference between the of the values in `x` and `max` where `where` is `True`.

    """
    max = max_tf(x, where=where, initial=initial)
    return delta_tf(x, detla=max, where=where)
