# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat import VarTransform
import pytest


def test_var():
    """
    Test variance transform.
    """

    x = np.random.rand(100)

    tf = VarTransform()
    y_hat = tf(x)

    y = np.var(x)

    assert pytest.approx(y_hat) == y
