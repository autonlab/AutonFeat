import numpy as np
from typing import Callable, Union

from autofeat.core import Transform
from autofeat.functional import kurtosis_tf


class KurtosisTransform(Transform):
    """
    Compute the kurtosis of the values in `x`.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self, name: str = "Kurtosis") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, fisher: Union[bool, np.bool_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[np.float_, np.int_]:
        """
        Compute the krutosis of the values in `x` where `where` is `True`.\n
        The krutosis is a measure of the "tailedness" of a distribution. It is defined as the fourth standardized moment of a distribution, and is calculated as:
        ```
        kurtosis = mean((x - mean(x)) / std(x)) ** 4
        ```

        Args:
            `signal_window`: The signal to compute the krutosis of.

            `fisher`: Whether to use Fisher's definition of kurtosis i.e. subtract 3 from the result. Default is `True`. If `False`, the result is the Pearson's definition of kurtosis.

            `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The krutosis of the values in `x` where `where` is `True`.
        """
        return kurtosis_tf(signal_window, fisher=fisher, where=where)
