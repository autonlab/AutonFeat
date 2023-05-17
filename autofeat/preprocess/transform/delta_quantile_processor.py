import numpy as np
from typing import Union, Callable
from autofeat.core import Preprocess
from autofeat.preprocess.functional import delta_quantile_tf


class DeltaQuantilePreprocessor(Preprocess):
    """
    Preprocess the signal by computing a delta (using `quantile`) with elements of a `signal`.

    Inherits from Preprocess.
    """
    # Dunder methods
    def __init__(self, name: str = "Delta Quantile") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, q: Union[float, np.float_], method: str = 'linear', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
        """
        Compute the difference between the values in `signal` and `quantile` where `where` is `True`.

        Args:
            `signal`: The array to compute the delta with.

            `q`: The quantile to compute. Must be between 0 and 1.

            `method`: The method to use when computing the quantile. Default is 'linear'. See `numpy.quantile` for more information.

            `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The difference between the of the values in `signal` and `quantile` where `where` is `True`.

        """
        return delta_quantile_tf(x=signal, q=q, method=method, where=where)
