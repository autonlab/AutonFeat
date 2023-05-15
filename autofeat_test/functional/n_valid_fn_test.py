import numpy as np
from autofeat.functional import n_valid_tf


def test_n_valid_fn():
    """
    Test n-valid functional transform.
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

        y_hat = n_valid_tf(x, where=invalid_fn)

        assert y_hat == n_values - n_invalid_values
