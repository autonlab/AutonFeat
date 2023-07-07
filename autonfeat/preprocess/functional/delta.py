import numpy as np
from typing import Union, Callable


def delta_tf(x: np.ndarray, delta: Union[int, float, np.int_, np.float_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
    """
    Preprocess the signal by shifting the `signal` by some `delta` value.

    Args:
        x: The signal to preprocess.

        delta: The value to shift the signal by.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The shifted signal.
    """
    # Vectorize where fn
    where_fn = np.vectorize(where)
    # Compute mask and multiply by distribution shift along axis
    mask = where_fn(x)
    shift = mask * delta
    # Compute shifted distribution
    return x - shift
