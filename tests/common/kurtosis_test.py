# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
import scipy.stats as stats
from autonfeat import KurtosisTransform
import pytest


def test_kurtosis_fisher():
    """
    Test kurtosis (Fisher) transform.
    """

    x = np.random.rand(100)
    tf = KurtosisTransform()

    y_hat = tf(x, fisher=True)

    y = stats.kurtosis(x, fisher=True)

    assert pytest.approx(y_hat) == y


def test_kurtosis_pearson():
    """
    Test kurtosis (Pearson) transform.
    """

    x = np.random.rand(100)

    tf = KurtosisTransform()
    y_hat = tf(x, fisher=False)

    y = stats.kurtosis(x, fisher=False)

    assert pytest.approx(y_hat) == y
