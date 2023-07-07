import numpy as np
from autofeat.preprocess.functional import delta_max_tf


def test_delta_max_fn():
    """
    Test max distribution shift functional form preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)

    shifted_x_hat = delta_max_tf(x)

    delta = np.max(x)
    shifted_x = x - delta

    assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
