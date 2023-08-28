# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat.functional import approx_entropy_tf
import pytest


def test_approximate_entropy_fn():
    """
    Test approximate entropy functional form transform.
    """

    x = np.random.rand(100)
    y_hat = approx_entropy_tf(x, m=3, r=0.2)

    y = 0.0

    # assert pytest.approx(y_hat) == y
