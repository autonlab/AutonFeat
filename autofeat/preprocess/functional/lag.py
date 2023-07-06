import numpy as np
from typing import Union, Callable
import warnings


def lag_tf(x: np.ndarray, lag: Union[int, np.int_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
    """
    Preprocess the signal by shifting it by a specified lag.

    Args:
        x: The signal to preprocess.

        lag: The lag to apply to the signal.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The signal lagged by the specified value.

    Warning:
        Only float arrays can be padded with NaNs. If the input array has a different dtype, the array elements will be cast to numpy float, if possible.
    """
    # Vectorize where fn
    where_fn = np.vectorize(where)
    filtered_x = x[where_fn(x)]

    # Only floats can be padded with NaNs
    if filtered_x.dtype != np.float_:
        warnings.warn("Only float arrays can be padded with NaNs but the input array has dtype {}. Trying to cast array elements to numpy float.".format(filtered_x.dtype))
        filtered_x = filtered_x.astype(np.float_)
    # Shift the signal elements by the specified lag
    shifted_x = np.roll(filtered_x, lag)
    # Pad the shifted signal with NaNs
    shifted_x[:lag] = np.nan
    return shifted_x
