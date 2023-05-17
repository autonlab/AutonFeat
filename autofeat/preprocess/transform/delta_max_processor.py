import numpy as np
from typing import Union, Callable
from autofeat.core import Preprocess
from autofeat.preprocess.functional import delta_max_tf


class DeltaMaxPreprocessor(Preprocess):
    """
    Preprocess the signal by computing a delta (using `max`) with elements of a `signal`.

    Inherits from Preprocess.
    """
    # Dunder methods
    def __init__(self, name: str = "Delta Max") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x), initial: Union[int, float, np.int_, np.float_] = -np.inf) -> np.ndarray:
        """
        Compute the difference between the values in `signal` and `max` where `where` is `True`.

        Args:
            `signal`: The array to compute the delta with.

            `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

            `initial`: The initial value for the maximum. Default is `-np.inf`.

        Returns:
            The difference between the of the values in `signal` and `max` where `where` is `True`.
        """
        return delta_max_tf(x=signal, where=where, initial=initial)
