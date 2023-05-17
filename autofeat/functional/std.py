import numpy as np
from typing import Union, Callable


def std_tf(x: np.ndarray, ddof: Union[int, np.int_] = 0, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the standard deviation of the values in `x`.

    Args:
        `x`: The array to compute the median of.

        `ddof`: The delta degrees of freedom. Default is `0`. See `numpy.std` for more information.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The standard deviation of the values in `x`.

    """
    return np.std(x, axis=0, ddof=ddof, where=[where(x_i) for x_i in x])
