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
from typing import Union, Callable, Optional


def entropy_tf(pk: np.ndarray, qk: Optional[np.ndarray] = None, base: Optional[Union[int, np.int_]] = None, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the entropy of the values in `pk` where `where` is `True`.

    Args:
        pk: The discrete probability distribution to find the entropy of.

        qk: The second discrete probability distribution to find the relative entropy with.

        base: The base of the logarithm used to compute the entropy. Default is `None` which means that the natural logarithm is used.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The entropy of the values in `pk` optionally with respect to `qk` (relative entropy) where `where` is `True`.
    """
    if base is not None and base <= 0:
        raise ValueError("Base must be a positive integer or `None`.")

    # Vectorize where fn
    where_fn = np.vectorize(pyfunc=where)

    # Get the valid values
    pk = pk[where_fn(pk)]
    qk = qk[where_fn(qk)] if qk is not None else None

    # Normalize distributions
    pk = pk / np.sum(pk, axis=0)
    qk = qk / np.sum(qk, axis=0) if qk is not None else None

    # Compute Shannon entropy or KL divergence
    S = -np.sum(pk * np.log(pk), axis=0) if qk is None else np.sum(pk * np.log(pk / qk), axis=0)

    # Correct units e.g. bits, nats, etc. with base
    if base is not None:
        S /= np.log(base)
    return S
