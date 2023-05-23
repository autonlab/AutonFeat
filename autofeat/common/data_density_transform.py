import numpy as np
from typing import Callable, Union

from autofeat.core import Transform
from autofeat.functional import data_density_tf


class DataDensityTransform(Transform):
    """
    Compute the number of valid measurements `x`.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self, name: str = "Data Density") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[np.float_, np.int_]:
        """
        Compute the data density of the array `x`.

        Args:
            `signal_window`: The signal window to find the data density of.

            `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            A scalar value representing the data density of `x`.
        """
        return data_density_tf(x=signal_window, where=where)
