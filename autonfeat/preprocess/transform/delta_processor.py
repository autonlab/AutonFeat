import numpy as np
from typing import Union, Callable
from autonfeat.core import Preprocess
from autonfeat.preprocess.functional import delta_tf


class DeltaPreprocessor(Preprocess):
    """
    Preprocess the signal by shifting the `signal` by some `delta` value.
    """
    # Dunder methods
    def __init__(self, name: str = "Delta") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, delta: Union[int, float, np.int_, np.float_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
        """
        Shift the `signal` by a `delta` where `where` is `True`.

        Args:
            signal: The array to shift.

            delta: The value to shift by.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The shifted signal.
        """
        return delta_tf(x=signal, delta=delta, where=where)
