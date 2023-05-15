import numpy as np
from typing import Union, Callable


def n_valid_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the number of valid measurements in `x` where `where` is `True` for valid measurements.

    Args:
        `x`: The array to compute the number of valid measurements in.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The number of valid measurements in `x`.

    """
    return np.sum(
        [
            int(where(x_i))
            for x_i in x
        ]
    )
