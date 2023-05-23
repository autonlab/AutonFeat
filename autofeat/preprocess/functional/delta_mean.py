import numpy as np
from typing import Union, Callable
from autofeat.functional import mean_tf
from autofeat.preprocess.functional import delta_tf


def delta_mean_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
    """
    Compute the difference between the values in `x` and `mean` where `where` is `True`.

    Args:
        `x`: The array to compute the delta with.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The difference between the of the values in `x` and `mean` where `where` is `True`.
    """
    mean = mean_tf(x, where=where)
    return delta_tf(x, delta=mean, where=where)
