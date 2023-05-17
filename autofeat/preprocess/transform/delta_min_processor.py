import numpy as np
from typing import Union, Callable
from autofeat.core import Preprocess
from autofeat.preprocess.functional import delta_min_tf


class DeltaMinPreprocessor(Preprocess):
    """
    Preprocess the signal by computing a delta (using `min`) with elements of a `signal`.

    Inherits from Preprocess.
    """
    # Dunder methods
    def __init__(self, name: str = "Delta Min") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x), initial: Union[int, float, np.int_, np.float_] = -np.inf) -> np.ndarray:
        """
        Compute the difference between the values in `signal` and `min` where `where` is `True`.

        Args:
            `signal`: The array to compute the delta with.

            `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

            `initial`: The initial value for the minimum. Default is `-np.inf`.

        Returns:
            The difference between the of the values in `signal` and `min` where `where` is `True`.
        """
        return delta_min_tf(x=signal, where=where, initial=initial)
