import numpy as np
from typing import Union, Callable


def delta_tf(x: np.ndarray, delta: Union[int, float, np.int_, np.float_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the difference between the values in `x` and `delta` where `where` is `True`.

    Args:
        `x`: The array to compute the delta with.

        `delta`: The value to compute the delta with.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The difference between the of the values in `x` and `delta` where `where` is `True`.

    """
    if where is None:
        return x - delta

    return np.array(
        [
            x_i - delta
            if where(x_i)
            else x_i
            for x_i in x
        ]
    )
