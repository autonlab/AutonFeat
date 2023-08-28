# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat.preprocess import DeltaPreprocessor


def test_delta():
    """
    Test delta distribution shift functional form preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)

    tf = DeltaPreprocessor()

    for _ in (10, 100, 10):
        random_delta = np.random.rand()

        shifted_x_hat = tf(x, delta=random_delta)

        shifted_x = x - random_delta

        assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
