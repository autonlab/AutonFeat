import numpy as np
from typing import Union, Callable
from autofeat.core import Preprocess
from autofeat.preprocess.functional import lag_tf


class LagPreprocessor(Preprocess):
    """
    Preprocess the signal by shifting the `signal` by some `delta` value.
    """
    # Dunder methods
    def __init__(self, name: str = "Lag") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, lag: Union[int, float, np.int_, np.float_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
        """
        Roll the `signal` by a `lag` where `where` is `True`. This pads the shifted signal with `NaN` values.

        Args:
            signal: The array to roll.

            lag: The lag to apply to the signal.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The shifted signal.
        """
        return lag_tf(x=signal, lag=lag, where=where)
