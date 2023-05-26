import numpy as np
from typing import Callable, Union

from autofeat.core import Transform
from autofeat.functional import sample_entropy_tf


class SampleEntropyTransform(Transform):
    """
    Compute the sample entropy of the signal.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self, name: str = "Sample Entropy") -> None:
        super().__init__(name=name)

    def __call__(signal_window: np.ndarray, m: Union[int, np.int_], r: Union[int, np.int_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
        """
        Compute the sample entropy of the values in `x` where `where` is `True`.\n
        This is a measure of the complexity of a signal. It is a modification of the approximate entropy (ApEn) algorithm. It can be computed with the formula:\n
        `SampEn = -log(A / B)`\n
        where `A` is the number of matches for template vectors of length `m` and `B` is the number of matches for template vectors of length `m + 1`.\n
        A match is defined as a template vector `xmi` that is close to another template vector `xmj` in the sense that the maximum absolute difference between their corresponding scalar elements is less than or equal to `r`.\n

        Args:
            `signal_window`: The signal to find the sample entropy of.

            `m`: The length of the template vector.

            `r`: The tolerance.

            `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The sample entropy of the values in `x` where `where` is `True`.
        """
        return sample_entropy_tf(x=signal_window, m=m, r=r, where=where)
