# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat import MedianTransform
import pytest


def test_median():
    """
    Test median transform.
    """

    x = np.random.rand(100)

    tf = MedianTransform()
    y_hat = tf(x)

    y = np.median(x)

    assert pytest.approx(y_hat) == y
