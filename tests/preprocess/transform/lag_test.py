# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

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
