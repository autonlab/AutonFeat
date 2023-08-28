# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat import QuantileTransform
import pytest


def test_quantile():
    """
    Test quantile transform.
    """

    x = np.random.rand(100)

    tf = QuantileTransform()

    for p in (10, 100, 10):
        y_hat = tf(x, p / 100)
        y = np.percentile(x, p)

        assert pytest.approx(y_hat) == y
