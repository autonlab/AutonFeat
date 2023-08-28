# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat import StdTransform
import pytest


def test_std():
    """
    Test standard deviation transform.
    """

    x = np.random.rand(100)

    tf = StdTransform()
    y_hat = tf(x)

    y = np.std(x)

    assert pytest.approx(y_hat) == y
