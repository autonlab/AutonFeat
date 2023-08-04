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
import numba as nb
from typing import Union, Callable


# Helper functions
def maxdist(x_i: np.ndarray, x_j: np.ndarray) -> Union[int, float, np.int_, np.float_]:
    """
    Compute the maximum distance between two vectors.

    Args:
        x_i: The first vector.

        x_j: The second vector.

    Returns:
        The maximum distance between the two vectors.
    """
    # Zip the vectors and compute the max distance
    zipped_x = np.array(list(zip(x_i, x_j)))
    distances = np.array([
        np.abs(zipped_x[i][0] - zipped_x[i][1])
        for i in nb.prange(len(zipped_x))
    ])
    return np.max(distances)


def phi(x_in, m, r, N):
    """
    Compute phi for a given vector x_in and the window length m.

    Args:
        x_in: The vector to compute phi for.

        m: The window length.

        r: The tolerance.

        N: The length of the vector.

    Returns:
        The phi value for the given vector x and window length m.
    """
    x = np.array([
        [x_in[j] for j in nb.prange(i, i + m - 1 + 1)]
        for i in nb.prange(N - m + 1)
    ])
    C = np.array([
        len([
            1
            for j in nb.prange(len(x))
            if maxdist(x_i=x[i], x_j=x[j]) <= r
        ]) / (N - m + 1.0)
        for i in nb.prange(len(x))
    ])
    return (N - m + 1.0) ** (-1) * np.sum(np.log(C))


# Approximate entropy function
def approx_entropy_tf(x: np.ndarray, m: Union[int, np.int_], r: Union[int, np.int_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the approximate entropy of the values in `x` where `where` is `True`. It used to quantify the amount of regularity and the unpredictability of fluctuations in the signal.

    Args:
        x: The signal to find the approximate entropy of.

        m: The length of the template vector.

        r: The tolerance.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The approximate entropy of the values in `x` where `where` is `True`.

    References:
        Approximate Entropy - https://en.wikipedia.org/wiki/Approximate_entropy
    """
    # Vectorize where fn
    where_fn = np.vectorize(pyfunc=where)
    # Get the valid values
    filtered_x = x[where_fn(x)]

    N = len(filtered_x)

    # Compute phi for m and m + 1
    phi_m = phi(x_in=filtered_x, m=m, r=r, N=N)
    phi_m_plus_1 = phi(x_in=filtered_x, m=m + 1, r=r, N=N)

    # Return the absolute difference as the approximate entropy
    return np.abs(phi_m_plus_1 - phi_m)
