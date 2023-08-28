# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat.preprocess import DeltaQuantilePreprocessor


def test_delta_quantile():
    """
    Test quantile distribution shift preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)

    tf = DeltaQuantilePreprocessor()

    for p in (10, 100, 10):
        shifted_x_hat = tf(x, q=p / 100)

        delta = np.percentile(x, p)
        shifted_x = x - delta

        assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
