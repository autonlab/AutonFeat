import numpy as np
from typing import Union

from autofeat.core import Transform
from autofeat.functional import iqr_tf


class IQRTransform(Transform):
    """
    Compute the inter-quartile range of the values.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self):
        super().__init__(name="IQR")

    def __call__(self, signal_window: np.ndarray, method: str = 'linear') -> Union[np.float_, np.int_]:
        """
        Compute the inter-quartile range of the values in `x`.

        Args:
            `x`: The array to compute the IQR of.

            `method`: The method to use when computing the quantiles. Default is 'linear'. See `numpy.quantile` for more information.

        Returns:
            A scalar value representing the IQR of the signal.

        """
        return iqr_tf(signal_window, method=method)
