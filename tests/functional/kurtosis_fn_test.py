# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
import scipy.stats as stats
from autonfeat.functional import kurtosis_tf
import pytest


def test_kurtosis_fisher_fn():
    """
    Test kurtosis (Fisher) functional form transform.
    """

    x = np.random.rand(100)

    y_hat = kurtosis_tf(x, fisher=True)

    y = stats.kurtosis(x, fisher=True)

    assert pytest.approx(y_hat) == y


def test_kurtosis_pearson_fn():
    """
    Test kurtosis (Pearson) functional form transform.
    """

    x = np.random.rand(100)

    y_hat = kurtosis_tf(x, fisher=False)

    y = stats.kurtosis(x, fisher=False)

    assert pytest.approx(y_hat) == y
