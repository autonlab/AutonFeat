# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat.functional import max_tf
import pytest


def test_max_fn():
    """
    Test max functional form transform.
    """

    x = np.random.rand(100)
    y_hat = max_tf(x)

    y = np.max(x)

    assert pytest.approx(y_hat) == y
