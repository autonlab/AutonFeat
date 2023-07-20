# MIT License

# Copyright (c) 2023 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from typing import Callable, Union

from autonfeat.core import Transform
from autonfeat.functional import kurtosis_tf


class KurtosisTransform(Transform):
    """
    Compute the kurtosis of the values in `x`.
    """
    # Dunder methods
    def __init__(self, name: str = "Kurtosis") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, fisher: Union[bool, np.bool_] = True, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[np.float_, np.int_]:
        """
        Compute the krutosis of the values in `x` where `where` is `True`.\n
        The krutosis is a measure of the "tailedness" of a distribution. It is defined as the fourth standardized moment of a distribution, and is calculated as:

        Args:
            signal_window: The signal to compute the krutosis of.

            fisher: Whether to use Fisher's definition of kurtosis i.e. subtract 3 from the result. Default is `True`. If `False`, the result is the Pearson's definition of kurtosis.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The krutosis of the values in `x` where `where` is `True`.
        """
        return kurtosis_tf(signal_window, fisher=fisher, where=where)
