import numpy as np
from autonfeat.preprocess import DeltaMedianPreprocessor


def test_delta_median():
    """
    Test median distribution shift preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)

    tf = DeltaMedianPreprocessor()
    shifted_x_hat = tf(x)

    delta = np.median(x)
    shifted_x = x - delta

    assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
