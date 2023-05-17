import numpy as np
from typing import Union, Callable


def quantile_tf(x: np.ndarray, q: Union[float, np.float_], method: str = 'linear', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the q-th quantile of the values in `x`.

    Args:
        `x`: The array to compute the q-th quantile of.

        `q`: The quantile to compute. `q` belongs to [0, 1].

        `method`: The method to use when computing the quantile. Default is 'linear'. See `numpy.quantile` for more information.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The q-th quantile of the values in `x`.

    Raises:
        `ValueError`: If `q` is not in [0, 1].

    """
    if q < 0 or q > 1:
        raise ValueError('q must be in [0, 1].')

    filtered_x = np.array([x_i for x_i in x if where(x_i)])
    return np.quantile(filtered_x, q, axis=0, method=method)
