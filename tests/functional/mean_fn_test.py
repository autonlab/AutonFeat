# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat.functional import mean_tf
import pytest


def test_mean_fn():
    """
    Test mean functional form transform.
    """

    x = np.random.rand(100)
    y_hat = mean_tf(x)
    y = 0.0
    for i in range(len(x)):
        y += x[i]
    y /= len(x)
    assert pytest.approx(y_hat) == y
