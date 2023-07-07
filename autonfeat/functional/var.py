import numpy as np
from typing import Union, Callable


def var_tf(x: np.ndarray, ddof: Union[int, np.int_] = 0, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the variance of the values in `x`.

    Args:
        x: The array to compute the variance of.

        ddof: The delta degrees of freedom. Default is `0`. See `numpy.var` for more information.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The variance of the values in `x`.
    """
    # Vectorize where fn
    where_fn = np.vectorize(pyfunc=where)
    return np.var(x, axis=0, ddof=ddof, where=where_fn(x))
