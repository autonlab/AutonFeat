# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat.functional import iqr_tf
import pytest


def test_iqr_fn():
    """
    Test inter-quartile range functional form transform.
    """

    x = np.random.rand(100)
    y_hat = iqr_tf(x)

    y = np.percentile(x, 75) - np.percentile(x, 25)

    assert pytest.approx(y_hat) == y
