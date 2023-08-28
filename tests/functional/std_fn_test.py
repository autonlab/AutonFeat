# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat.functional import std_tf
import pytest


def test_std_fn():
    """
    Test standard deviation functional form transform.
    """

    x = np.random.rand(100)
    y_hat = std_tf(x)

    y = np.std(x)

    assert pytest.approx(y_hat) == y
