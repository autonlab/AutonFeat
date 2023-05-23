import numpy as np
from typing import Union, Callable


def delta_tf(x: np.ndarray, delta: Union[int, float, np.int_, np.float_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
    """
    Compute the difference between the values in `x` and `delta` where `where` is `True`.

    Args:
        `x`: The array to compute the delta with.

        `delta`: The value to compute the delta with.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The difference between the of the values in `x` and `delta` where `where` is `True`.
    """
    # Vectorize where fn
    where_fn = np.vectorize(where)
    # Compute mask and multiply by distribution shift along axis
    mask = where_fn(x)
    shift = mask * delta
    # Compute shifted distribution
    return x - shift
