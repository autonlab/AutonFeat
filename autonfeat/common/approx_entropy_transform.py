# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Callable, Union

from autonfeat.core import Transform
from autonfeat.functional import approx_entropy_tf


class ApproxEntropyTransform(Transform):
    """
    Compute the approximate entropy of the signal.

    References:
        Approximate Entropy - https://en.wikipedia.org/wiki/Approximate_entropy
    """
    # Dunder methods
    def __init__(self, name: str = "Approximate Entropy") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, m: Union[int, np.int_], r: Union[int, np.int_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
        """
        Compute the approximate entropy of the values in `x` where `where` is `True`. It used to quantify the amount of regularity and the unpredictability of fluctuations in the signal.

        Args:
            signal_window: The signal to find the approximate entropy of.

            m: The length of the template vector.

            r: The tolerance.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The approximate entropy of the values in `x` where `where` is `True`.
        """
        return approx_entropy_tf(x=signal_window, m=m, r=r, where=where)
