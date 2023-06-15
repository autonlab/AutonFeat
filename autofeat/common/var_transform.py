import numpy as np
from typing import Callable, Union

from autofeat.core import Transform
from autofeat.functional import var_tf


class VarTransform(Transform):
    """
    Compute the variance of the values in `x`.
    """
    # Dunder methods
    def __init__(self, name: str = "Variance") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, ddof: Union[int, np.int_] = 0, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[np.float_, np.int_]:
        """
        Compute the variance of the signal window provided.

        Args:
            signal_window: The signal window to find the variance of.

            ddof: The delta degrees of freedom. Default is `0`. See `numpy.var` for more information.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            A scalar value representing the variance of the signal.
        """
        return var_tf(signal_window, ddof=ddof, where=where)
