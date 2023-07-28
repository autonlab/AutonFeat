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
from autonfeat.preprocess import LagPreprocessor


def nan_compare(arr1, arr2):
    """
    Compare two arrays.
    """
    assert len(arr1) == len(arr2)
    for i in range(len(arr1)):
        if np.isnan(arr1[i]) and np.isnan(arr2[i]):
            continue
        elif arr1[i] == arr2[i]:
            continue
        else:
            return False
    return True


def test_lag():
    """
    Test the lag preprocessor.
    """
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    preprocessor = LagPreprocessor()

    # Test 1
    y = np.array([np.nan, np.nan, 1, 2, 3, 4, 5, 6])
    y_hat = preprocessor(x, lag=2)
    assert nan_compare(y_hat, y)

    # Test 2
    y = np.array([np.nan, np.nan, np.nan, 1, 2, 3, 4, 5])
    y_hat = preprocessor(x, lag=3)
    assert nan_compare(y_hat, y)

    # Test 3
    y = np.array([3, 4, 5, 6, 7, 8, np.nan, np.nan])
    y_hat = preprocessor(x, lag=-2)
    assert nan_compare(y_hat, y)

    # Test 4
    y = np.array([4, 5, 6, 7, 8, np.nan, np.nan, np.nan])
    y_hat = preprocessor(x, lag=-3)
    assert nan_compare(y_hat, y)
