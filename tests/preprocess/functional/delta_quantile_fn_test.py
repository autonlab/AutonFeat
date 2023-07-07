import numpy as np
from autonfeat.preprocess.functional import delta_quantile_tf


def test_delta_quantile_fn():
    """
    Test quantile distribution shift functional form preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)

    for p in (10, 100, 10):
        shifted_x_hat = delta_quantile_tf(x, q=p / 100)

        delta = np.percentile(x, p)
        shifted_x = x - delta

        assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
