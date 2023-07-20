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
from typing import Union, Callable
import warnings


def lag_tf(x: np.ndarray, lag: Union[int, np.int_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
    """
    Preprocess the signal by shifting it by a specified lag.

    Args:
        x: The signal to preprocess.

        lag: The lag to apply to the signal.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The signal lagged by the specified value.

    Warning:
        Only float arrays can be padded with NaNs. If the input array has a different dtype, the array elements will be cast to numpy float, if possible.
    """
    # Vectorize where fn
    where_fn = np.vectorize(where)
    filtered_x = x[where_fn(x)]

    # Only floats can be padded with NaNs
    if filtered_x.dtype != np.float_:
        warnings.warn("Only float arrays can be padded with NaNs but the input array has dtype {}. Trying to cast array elements to numpy float.".format(filtered_x.dtype))
        filtered_x = filtered_x.astype(np.float_)
    # Shift the signal elements by the specified lag
    shifted_x = np.roll(filtered_x, lag)
    # Pad the shifted signal with NaNs
    if lag < 0:
        shifted_x[lag:] = np.nan
    else:
        shifted_x[:lag] = np.nan
    return shifted_x
