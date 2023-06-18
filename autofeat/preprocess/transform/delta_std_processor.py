import numpy as np
from typing import Union, Callable
from autofeat.core import Preprocess
from autofeat.preprocess.functional import delta_std_tf


class DeltaStdPreprocessor(Preprocess):
    """
    Preprocess the signal by shifting each element in the signal by the standard deviation of the signal.
    """
    # Dunder methods
    def __init__(self, name: str = "Delta Standard Deviation") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, ddof: Union[int, np.int_] = 0, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
        """
        Compute the standard deviation of the signal and shift the signal by this standard deviation.

        Args:
            signal: The array to compute the delta with.

            ddof: The delta degrees of freedom. Default is `0`. See `numpy.std` for more information.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The shifted signal.
        """
        return delta_std_tf(x=signal, ddof=ddof, where=where)
