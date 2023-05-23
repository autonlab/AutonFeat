import numpy as np
from typing import Union, Callable
from autofeat.core import Preprocess
from autofeat.preprocess.functional import delta_std_tf


class DeltaStdPreprocessor(Preprocess):
    """
    Preprocess the signal by computing a delta (using `std`) with elements of a `signal`.

    Inherits from Preprocess.
    """
    # Dunder methods
    def __init__(self, name: str = "Delta Standard Deviation") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, ddof: Union[int, np.int_] = 0, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
        """
        Compute the difference between the values in `signal` and `std` where `where` is `True`.

        Args:
            `signal`: The array to compute the delta with.

            `ddof`: The delta degrees of freedom. Default is `0`. See `numpy.std` for more information.

            `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The difference between the of the values in `signal` and `std` where `where` is `True`.
        """
        return delta_std_tf(x=signal, ddof=ddof, where=where)
