# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from autonfeat import DataDensityTransform


def test_data_density():
    """
    Test data density transform.
    """
    n_values = 100
    n_invalid_values = 10
    invalid_values = [
        np.nan,
        np.inf,
    ]

    invalid_fns = [
        lambda x: not np.isnan(x),
        lambda x: not np.isinf(x),
    ]

    for invalid_value, invalid_fn in zip(invalid_values, invalid_fns):
        x = np.random.rand(n_values)

        # Random n_invalid values
        invalid_idx = np.random.choice(
            np.arange(n_values),
            size=n_invalid_values,
            replace=False,
        )
        x[invalid_idx] = invalid_value

        tf = DataDensityTransform()
        y_hat = tf(x, where=invalid_fn)

        assert y_hat == (n_values - n_invalid_values) / n_values
