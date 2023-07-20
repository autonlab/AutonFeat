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


def dft_tf(x: np.ndarray, n: Union[int, np.int_] = None, norm: str = 'backward', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the 1D discete Fourier Transform (DFT) using Fast-Fourier Transform (FFT) on the values in `x`.

    Args:
        x: The array to compute the DFT of.

        n: The number of points to use for the FFT. If `None`, the length of `x` is used. Default is `None`.

        norm: The normalization mode to use. Default is 'backward'. See `numpy.fft` for more information.
                    Options include:\n
                    'backward': The backward transform is scaled by `1/n`.
                    'ortho': The forward and backward transforms are scaled by `1/sqrt(n)`.
                    'forward': The forward transform is not scaled.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The 1D DFT of `x`.  It should be noted that the result contains complex numbers. The absolute value of the result can be considered for further processing.
    """
    # Vectorize the where function
    where_fn = np.vectorize(where)
    filtered_x = x[where_fn(x)]
    return np.fft.fft(a=filtered_x, n=n, norm=norm, axis=0)
