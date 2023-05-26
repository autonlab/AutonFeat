import numpy as np
from typing import Callable, Union

from autofeat.core import Transform
from autofeat.functional import skewness_tf


class SkewnessTransform(Transform):
    """
    Compute the skewness of the values in `x`.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self, name: str = "Skewness") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[np.float_, np.int_]:
        """
        Compute the skewness of the values in `x` where `where` is `True`.
        The skewness is computed using the Fisher-Pearson standardized coefficient of skewness.\n
        The sample skewness is computed as:
        ```
        g_1 = m_3 / m_2^(3/2)
        ```
        where `m_2` is the second central moment and `m_3` is the third central moment.\n
        The Fisher-Pearson standardized coefficient of skewness is computed as:
        ```
        G_1 = g_1 * sqrt(n(n-1)) / (n-2)
        ```

        Note:
            The skewness is only computed for valid values i.e. values where `where` is `True`. The skewness computed is corrected for statistical bias.

        Args:
            `signal_window`: The signal to compute the skewness of.

            `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The skewness of the values in `x` where `where` is `True`.
        """
        return skewness_tf(signal_window, where=where)
