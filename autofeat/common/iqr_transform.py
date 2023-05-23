import numpy as np
from typing import Union, Callable

from autofeat.core import Transform
from autofeat.functional import iqr_tf


class IQRTransform(Transform):
    """
    Compute the inter-quartile range of the values.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self, name: str = "IQR") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, method: str = 'linear', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[np.float_, np.int_]:
        """
        Compute the inter-quartile range of the values in `x`.

        Args:
            `x`: The array to compute the IQR of.

            `method`: The method to use when computing the quantiles. Default is 'linear'. See `numpy.quantile` for more information.

            `where`: `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            A scalar value representing the IQR of the signal.
        """
        return iqr_tf(signal_window, method=method, where=where)
