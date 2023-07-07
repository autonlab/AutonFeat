import numpy as np
from autofeat.preprocess import DeltaMinPreprocessor


def test_delta_min():
    """
    Test min distribution shift preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)

    tf = DeltaMinPreprocessor()
    shifted_x_hat = tf(x)

    delta = np.min(x)
    shifted_x = x - delta

    assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
