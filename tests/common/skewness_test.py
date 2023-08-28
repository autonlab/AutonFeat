# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
import scipy.stats as stats
from autonfeat import SkewnessTransform
import pytest


def test_skewness():
    """
    Test skewness transform.
    """

    x = np.random.rand(100)

    tf = SkewnessTransform()
    y_hat = tf(x)

    y = stats.skew(x, bias=False)

    assert pytest.approx(y_hat) == y
