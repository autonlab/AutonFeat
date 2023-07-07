import numpy as np
from typing import Union, Callable
from autonfeat.functional import max_tf
from autonfeat.preprocess.functional import delta_tf


def delta_max_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x), initial: Union[int, float, np.int_, np.float_] = -np.inf) -> np.ndarray:
    """
    Preprocess the signal `x` by shifting each element of `x` by the maximum of `x`.

    Args:
        x: The array to compute the difference from its maximum.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        initial: The initial value for the maximum. Default is `-np.inf`.

    Returns:
        The shifted signal.
    """
    max = max_tf(x, where=where, initial=initial)
    return delta_tf(x, delta=max, where=where)
