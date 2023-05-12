import numpy as np
from typing import Union
from autofeat.functional import quantile_tf


def iqr_tf(x: np.ndarray, method: str = 'linear') -> Union[float, np.float_]:
    """
    Compute the inter-quartile range of the values in `x`.

    Args:
        `x`: The array to compute the IQR of.

        `method`: The method to use when computing the quantiles. Default is 'linear'. See `numpy.quantile` for more information.

    Returns:
        The IQR of the values in `x`.

    """

    return quantile_tf(x, q=0.75, method=method) - quantile_tf(x, q=0.25, method=method)
