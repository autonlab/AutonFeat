import numpy as np
from autofeat.preprocess.functional import delta_std_tf


def test_delta_std_fn():
    """
    Test standard deviation distribution shift functional form preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)
    shifted_x_hat = delta_std_tf(x)

    delta = np.std(x)
    shifted_x = x - delta

    assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
