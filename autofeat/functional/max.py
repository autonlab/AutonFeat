import numpy as np
from typing import Union, Callable


def max_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x), initial: Union[int, float, np.int_, np.float_] = -np.inf) -> Union[float, np.float_]:
    """
    Compute the max of the values in `x` where `where` is `True`.

    Args:
        `x`: The array to compute the max of.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        `initial`: The initial value to use when computing the max. Default is `-np.inf`.

    Returns:
        The max of the values in `x` where `where` is `True`.

    """
    return np.amax(x, axis=0, where=[where(x_i) for x_i in x], initial=initial)
