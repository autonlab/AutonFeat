# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat import MaxTransform
import pytest


def test_max():
    """
    Test max transform.
    """

    x = np.random.rand(100)

    tf = MaxTransform()
    y_hat = tf(x)

    y = np.max(x)

    assert pytest.approx(y_hat) == y
