import numpy as np
from typing import Callable, Union

from autofeat.core import Transform
from autofeat.functional import n_valid_tf


class NValidTransform(Transform):
    """
    Compute the number of valid measurements `x`.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self, name: str = "N-Valid") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[np.float_, np.int_]:
        """
        Compute the number of valid measurements in `x` where `where` is `True` for valid measurements.

        Args:
            `signal_window`: The signal window to find number of valid measurements in.

            `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            A scalar value representing the number of valid measurements in `x`.
        """
        return n_valid_tf(signal_window, where=where)
