import numpy as np
from typing import Union, Callable
from autonfeat.functional import quantile_tf
from autonfeat.preprocess.functional import delta_tf


def delta_quantile_tf(x: np.ndarray, q: Union[float, np.float_], method: str = 'linear', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
    """
    Preprocess the signal `x` by shifting each element of `x` by a quantile of `x`.

    Args:
        x: The array to shift by its quantile.

        q: The quantile to compute. Must be between 0 and 1.

        method: The method to use when computing the quantile. Default is 'linear'. See `numpy.quantile` for more information.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The shifted signal.
    """
    quantile = quantile_tf(x, q=q, method=method, where=where)
    return delta_tf(x, delta=quantile, where=where)
