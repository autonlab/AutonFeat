import numpy as np
from typing import Union, Callable


def mean_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the mean of the values in `x` where `where` is `True`.

    Args:
        `x`: The array to compute the mean of.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The mean of the values in `x` where `where` is `True`.
    """
    # Vectorize where fn
    where_fn = np.vectorize(pyfunc=where)
    return np.mean(x, where=where_fn(x))
