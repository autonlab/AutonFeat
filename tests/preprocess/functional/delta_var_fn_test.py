# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat.preprocess.functional import delta_var_tf


def test_delta_var_fn():
    """
    Test variance distribution shift functional form preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)
    shifted_x_hat = delta_var_tf(x)

    delta = np.var(x)
    shifted_x = x - delta

    assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
