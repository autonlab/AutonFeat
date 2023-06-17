import numpy as np
from typing import Union, Callable
from autofeat.core import Preprocess
from autofeat.preprocess.functional import delta_mean_tf


class DeltaMeanPreprocessor(Preprocess):
    """
    Preprocess the signal by computing a delta (using `mean`) with elements of a `signal` and shifting the `signal` by this delta.
    """
    # Dunder methods
    def __init__(self, name: str = "Delta Mean") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
        """
        Compute the difference between the values in `signal` and `mean` where `where` is `True`.

        Args:
            signal: The array to compute the delta with.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The shifted signal.
        """
        return delta_mean_tf(x=signal, where=where)
