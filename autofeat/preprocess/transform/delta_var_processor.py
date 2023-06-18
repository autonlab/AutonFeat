import numpy as np
from typing import Union, Callable
from autofeat.core import Preprocess
from autofeat.preprocess.functional import delta_var_tf


class DeltaVarPreprocessor(Preprocess):
    """
    Preprocess the signal by shifting each element in the signal by the variance of the signal.
    """
    # Dunder methods
    def __init__(self, name: str = "Delta Variance") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, ddof: Union[int, np.int_] = 0, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
        """
        Compute the variance of the signal and shift the signal by this variance.

        Args:
            signal: The array to compute the delta with.

            ddof: The delta degrees of freedom. Default is `0`. See `numpy.var` for more information.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The shifted signal.
        """
        return delta_var_tf(x=signal, ddof=ddof, where=where)
