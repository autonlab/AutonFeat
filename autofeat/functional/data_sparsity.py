import numpy as np
from typing import Union, Callable
from autofeat.functional import n_valid_tf


def data_sparsity_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the data sparsity of the array `x`.

    Args:
        `x`: The array to compute the data sparsity of.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The data sparsity of measurements in `x`.

    Raises:
        `DivideByZeroError`: If `x` is empty.
    """
    size = x.shape[0]
    if size == 0:
        raise ZeroDivisionError("Cannot compute data sparsity of empty array.")

    return (size - n_valid_tf(x=x, where=where)) / size
