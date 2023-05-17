import numpy as np
from typing import Union, Callable
from autofeat.core import Preprocess
from autofeat.preprocess.functional import delta_tf


class DeltaPreprocessor(Preprocess):
    """
    Preprocess the signal by computing a `delta` with elements of a `signal`.

    Inherits from Preprocess.
    """
    # Dunder methods
    def __init__(self, name: str = "Delta") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, delta: Union[int, float, np.int_, np.float_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
        """
        Compute the difference between the values in `signal` and `delta` where `where` is `True`.

        Args:
            `signal`: The array to compute the delta with.

            `delta`: The value to compute the delta with.

            `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The difference between the of the values in `signal` and `delta` where `where` is `True`.
        """
        return delta_tf(x=signal, delta=delta, where=where)
