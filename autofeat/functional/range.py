import numpy as np
from typing import Union, Callable
from autofeat.functional import max_tf, min_tf


def range_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = None) -> Union[float, np.float_]:
    """
    Compute the range of the values in `x`.

    Args:
        `x`: The array to compute the range of.

        `where`: A function that takes a value and returns `True` or `False` for whether it is to be included in the computation. Default is `None`.

    Returns:
        The range of the values in `x`.

    """

    return max_tf(x, where=where) - min_tf(x, where=where)
