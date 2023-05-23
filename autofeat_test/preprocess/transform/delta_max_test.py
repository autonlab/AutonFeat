import numpy as np
from autofeat.preprocess.transform import DeltaMaxPreprocessor


def test_delta_max():
    """
    Test max distribution shift preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)

    tf = DeltaMaxPreprocessor()
    shifted_x_hat = tf(x)

    delta = np.max(x)
    shifted_x = x - delta

    assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
