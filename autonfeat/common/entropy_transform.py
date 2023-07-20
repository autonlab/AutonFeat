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
from typing import Callable, Union, Optional

from autonfeat.core import Transform
from autonfeat.functional import entropy_tf


class EntropyTransform(Transform):
    """
    Compute the entropy of a distribution, or the KL divergence between two distributions.
    """
    # Dunder methods
    def __init__(self, name: str = "Shannon Entropy/KL Divergence") -> None:
        super().__init__(name=name)

    def __call__(self, pk: np.ndarray, qk: Optional[np.ndarray] = None, base: Optional[Union[int, np.int_]] = None, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
        """
        Compute the entropy of the values in `pk` where `where` is `True`.

        Args:
            pk: The discrete probability distribution to find the entropy of.

            qk: The second discrete probability distribution to find the relative entropy with. Default is `None`.

            base: The base of the logarithm used to compute the entropy. Default is `None` which means that the natural logarithm is used.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The entropy of the values in `pk` optionally with respect to `qk` (relative entropy) where `where` is `True`.
        """
        return entropy_tf(pk, qk=qk, base=base, where=where)
